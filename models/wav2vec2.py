from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from packaging import version
from transformers import Wav2Vec2Model as _Wav2Vec2ModelBase
from transformers.modeling_outputs import BaseModelOutput

# The implementation of Wav2Vec2Model is borrowed from
# https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html
# We initialize the encoder with pre-trained wav2vec 2.0 weights.


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([
            mask_idc[j] + offset
            for j in range(len(mask_idc))
            for offset in range(lengths[j])
        ])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


def linear_interpolation(
    features: torch.Tensor,
    input_fps: int,
    output_fps: int,
    output_len: int | None = None,
) -> torch.Tensor:
    """Linearly interpolate features from input_fps to output_fps.

    Args:
        features: (N, C, L)
        input_fps: source frame rate
        output_fps: target frame rate
        output_len: optional explicit output length

    Returns:
        torch.Tensor: (N, C, output_len)
    """
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    return F.interpolate(features, size=output_len, align_corners=False, mode='linear')


class Wav2Vec2Model(_Wav2Vec2ModelBase):
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.is_old_version = version.parse(transformers.__version__) < version.parse('4.7.0')

    def forward(
        self,
        input_values: torch.Tensor,
        output_fps: int = 25,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        frame_num: int | None = None,
    ) -> BaseModelOutput | tuple:
        self.config.output_attentions = True
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.feature_extractor(input_values)  # (N, C, L)
        # Resample audio features @ 50 fps to `output_fps`
        if frame_num is not None:
            hidden_states_len = round(frame_num * 50 / output_fps)
            hidden_states = hidden_states[:, :, :hidden_states_len]
        hidden_states = linear_interpolation(hidden_states, 50, output_fps, output_len=frame_num)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, C)

        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            attention_mask = torch.zeros(
                hidden_states.shape[:2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_mask[
                (
                    torch.arange(attention_mask.shape[0], device=hidden_states.device),
                    output_lengths - 1,
                )
            ] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        if self.is_old_version:
            hidden_states = self.feature_projection(hidden_states)
        else:
            hidden_states = self.feature_projection(hidden_states)[0]

        if self.config.apply_spec_augment and self.training:
            batch_size, sequence_length, hidden_size = hidden_states.size()
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(
                    hidden_states.dtype
                )
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
