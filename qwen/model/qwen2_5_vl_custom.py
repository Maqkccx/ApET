from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np
import einops as ein

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    PatchMerger,
    Qwen2RMSNorm,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    VisionAttention,
    VisionRotaryEmbedding,
    VisionSdpaAttention,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from qwen.model.qwen2_5_vl_utils import apply_rotary_pos_emb_vision, rotate_half, token_merging, \
    repeat_kv, apply_multimodal_rotary_pos_emb, apply_rotary_pos_emb_flashatt, cluster_and_merge, fps

from flash_attn import flash_attn_func, flash_attn_varlen_func

import sys
import time


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    pass


class Qwen2_5_VLForConditionalGeneration_X(Qwen2VLForConditionalGeneration):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        start_time = time.time()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # RoPE for pre-fill stage
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                pass

        if inputs_embeds is None:
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]  # [n, 3584]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = (input_ids == self.config.image_token_id)
                # print(n_image_tokens)

                # print(position_ids.shape)
                # print("x1:", position_ids[0,0,:])
                # print("x2:", position_ids[1,0,:])
                # print("x3:", position_ids[2,0,:])

                self.model.keep_indices = None
                self.model.image_grid_thw = image_grid_thw
                visual_token_ratio = self.image_token_ratio
                if visual_token_ratio != 1:
                    num_keep_tokens = int(visual_token_ratio * n_image_tokens)
                    indices = torch.nonzero(mask)
                    k = self.model.basis_token_num

                    # DPC
                    # seed_embeds = cluster_and_merge(image_embeds, k)

                    # FPS
                    seed_embeds = image_embeds.unsqueeze(0).gather(1, fps(image_embeds.unsqueeze(0), k).unsqueeze(-1).expand(-1,-1,image_embeds.shape[-1]))
                    seed_embeds = seed_embeds.squeeze()

                    seed_embeds = seed_embeds.float()
                    image_embeds = image_embeds.float()
                    G = torch.matmul(seed_embeds, seed_embeds.t())
                    G.diagonal().add_(1e-5)
                    linear_group = torch.linalg.solve(G, torch.matmul(seed_embeds, image_embeds.t())).t()  # [N, k]
                    image_embeds_hat = torch.matmul(linear_group, seed_embeds)  # [N, d]
                    errors = torch.norm(image_embeds - image_embeds_hat, dim=-1)  # [N]
                    keep_indices = torch.topk(errors, k=num_keep_tokens, dim=0)[1]
                    image_embeds = image_embeds.bfloat16()
                    seed_embeds = seed_embeds.bfloat16()

                    index_mask = torch.zeros(n_image_tokens, dtype=torch.bool, device=image_embeds.device)
                    index_mask.scatter_(0, keep_indices[-k:], True)
                    image_embeds[index_mask] = 0.5 * seed_embeds + 0.5 * image_embeds[index_mask]


                    # Directly drop other tokens
                    # image_embeds = image_embeds[keep_indices, :]
                    # Token merging
                    image_embeds = token_merging(image_embeds, keep_indices, k)

                    keep_indices = torch.sort(keep_indices, dim=0)[0]
                    self.model.keep_indices = keep_indices

                    image_embeds = image_embeds[keep_indices]
                    # image_embeds = torch.cat([image_embeds, seed_embeds], dim=0)

                    # Select the tokens with the lowest attention weights to remove
                    all_indices = torch.arange(n_image_tokens).to(keep_indices.device)
                    remove_indices = all_indices[~torch.isin(all_indices, keep_indices)]
                    indices_to_remove = indices[remove_indices]
                    # indices_to_remove = indices[:num_remove_tokens]
                    remove_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    for index in indices_to_remove:
                        remove_mask[index[0], index[1]] = False
                    input_ids = input_ids[remove_mask].reshape(input_ids.shape[0], -1)
                    # Correctly apply the mask for position ids across all heads
                    position_ids = position_ids[remove_mask.unsqueeze(0).expand(position_ids.shape[0], -1, -1)].reshape(
                        position_ids.shape[0], position_ids.shape[1], -1)

                n_image_tokens_after = (input_ids == self.config.image_token_id).sum().item()
                n_image_features_after = image_embeds.shape[0]
                self.model.n_image_tokens = n_image_tokens_after
                self.model.image_start_index = torch.nonzero(mask)[0, 1]
                if n_image_tokens_after != n_image_features_after:
                    raise ValueError(
                        f"Image features and image tokens do not match after pruning: tokens: {n_image_tokens_after}, features {n_image_features_after}"
                    )
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_mask = image_mask.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)
                self.model.n_image_tokens = 0

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds, attn_weights = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # batch_size, seq_length, _ = inputs_embeds.shape
            # attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                pass
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        end_time = time.time()
        prefill_time = end_time - start_time
        if hidden_states.shape[1] != 1:  # prefill stage
            seq_len_now = hidden_states.shape[1]
            print(f"[Prefill] current sample:"
                  f"seq_len={seq_len_now}  time={prefill_time * 1000:.2f} ms")
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen2_5_VLPreTrainedModel(Qwen2VLPreTrainedModel):
    pass


class Qwen2_5_VLModel_X(Qwen2_5_VLPreTrainedModel):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        start_time = time.time()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # cache_position = None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)


        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )


        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        cur_image_tokens = self.n_image_tokens

        sum_visual_attention = []
        # 28 x Qwen2_5_VLDecoderLayer
        # print(self.layers)
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Modify here
            # rank & drop after specific layer
            # only drop in prefill stage when inference
            image_token_ratio_list = self.image_token_ratio_list
            rank_layer = layer_idx + 1
            # if rank_layer in self.layer_list:
            #     if hidden_states.shape[1] != 1:  # prefill stage
            #         if cur_image_tokens > 0:
            #             stage = self.layer_list.index(rank_layer)  # determine current stage
            #             next_image_tokens = int(image_token_ratio_list[stage] * cur_image_tokens)
            #             (
            #                 position_ids,
            #                 attention_mask,
            #                 hidden_states
            #             ) = self.layer_prune(
            #                 cur_num=stage,
            #                 rank_layer=rank_layer,
            #                 features=hidden_states,
            #                 position_ids=position_ids,
            #                 attention_mask=causal_mask,
            #                 position_embeddings=position_embeddings,
            #                 cur_image_tokens=cur_image_tokens,
            #                 next_image_tokens=next_image_tokens,
            #             )
            #             position_embeddings = self.rotary_emb(hidden_states, position_ids)
            #             cur_image_tokens = next_image_tokens


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        end_time = time.time()
        # prefill_time = end_time - start_time
        # if hidden_states.shape[1] != 1:  # prefill stage
        #     seq_len_now = hidden_states.shape[1]
        #     print(f"[Prefill] current sample:"
        #           f"seq_len={seq_len_now}  time={prefill_time * 1000:.2f} ms")
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def layer_prune(
            self, cur_num, rank_layer, features, position_ids, attention_mask, position_embeddings, cur_image_tokens,
            next_image_tokens
    ):

        _position_ids = position_ids
        _attention_mask = attention_mask

        # print(features.shape) # [1, 357, 3584]
        # print(position_ids.shape) # [3, 1, 357]
        # print(attention_mask) # [1, 357, 3584]

        batch_size = features.shape[0]  # 1
        seq_len = position_ids.shape[2]  # 357
        image_start_index = self.image_start_index
        image_end_index = image_start_index + cur_image_tokens
        image_embeds = features[0][image_start_index:image_end_index, :]
        # k = int(math.ceil(next_image_tokens * 0.1)) + 1
        k = 10

        # Random
        # index_mask = torch.zeros(cur_image_tokens, dtype=torch.bool, device=image_embeds.device)
        # index_mask[:k] = True
        # rand_idx = torch.randint(0, cur_image_tokens, (k,), device=image_embeds.device)
        # rand_idx = rand_idx.unsqueeze(-1).expand(-1, image_embeds.size(-1))
        # seed_features = image_embeds.gather(0, rand_idx)

        # DPC
        # seed_features = cluster_and_merge(image_embeds, k)

        # FPS
        seed_features = image_embeds.unsqueeze(0).gather(1, fps(image_embeds.unsqueeze(0), k).unsqueeze(-1).expand(-1, -1,
                                                                                                                 image_embeds.shape[
                                                                                                                  -1]))
        seed_features = seed_features.squeeze()

        # seed_features = image_embeds[index_mask]

        image_embeds = image_embeds.float()
        seed_features = seed_features.float()
        G = torch.matmul(seed_features, seed_features.t())  # [k, k]
        G.diagonal().add_(1e-1)
        linear_group = torch.linalg.solve(G, torch.matmul(seed_features, image_embeds.t())).t()  # [N, k]
        image_embeds_hat = torch.matmul(linear_group, seed_features)  # [N, d]
        errors = torch.norm(image_embeds - image_embeds_hat, dim=-1)  # [N]
        seed_features = seed_features.bfloat16()
        image_embeds = image_embeds.bfloat16()
        top_rank_index = torch.topk(errors, k=next_image_tokens, dim=0)[1]

        # image_start_index = image_start_index.to(top_rank_index.device)
        # top_rank_index = top_rank_index + image_start_index

        image_embeds[top_rank_index[-k:], :] = 0.5 * seed_features + 0.5 * image_embeds[top_rank_index[-k:], :]
        image_embeds = token_merging(image_embeds, top_rank_index, k, scaling=1)
        top_rank_index = top_rank_index.sort().values

        image_embeds = image_embeds[top_rank_index]
        start_index = image_end_index
        new_input_embeds = torch.cat([features[0, :image_start_index, :], image_embeds, features[0, start_index:, :]], dim=0)
        # new_input_embeds = torch.cat(
        #     [features[0, :image_start_index, :], features[0, top_rank_index, :], features[0, start_index:, :]], dim=0)
        new_input_embeds = new_input_embeds.unsqueeze(0)
        top_rank_index = top_rank_index + image_start_index
        top_rank_index = top_rank_index.to(position_ids.device)
        start_index = start_index.to(position_ids.device)
        new_position_ids = torch.cat([position_ids[:, :, :image_start_index], position_ids[:, :, top_rank_index],
                                      position_ids[:, :, start_index:]], dim=2)

        # new_attention_mask = torch.cat([attention_mask[i][:image_index], attention_mask[i][errors_indices],
        #                                 attention_mask[i][text_index:]], dim=0)

        if _position_ids is None:
            position_ids = None

        return new_position_ids, attention_mask, new_input_embeds
