# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional

import torch
from seamless_communication.models.unity.model import UnitYModel, UnitYNART2UModel
from seamless_communication.models.unity.unit_tokenizer import UnitTokenizer
from seamless_communication.streaming.agents.online_text_decoder import (
    UnitYTextDecoderOutput,
)
from seamless_communication.streaming.agents.common import AgentStates
from simuleval.agents import GenericAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import Segment, TextSegment

import copy

from config.config import ControlSwitch
control_switch = ControlSwitch()

class NARUnitDecoderAgentStates(AgentStates):  # type: ignore
    def reset(self) -> None:
        self.source_token_list: List[str] = []
        self.source_indices: Optional[torch.Tensor] = None
        self.duration_start_index: int = 0
        self.tgt_lang = None
        super().reset()

    def update_source(self, segment: Segment) -> None:
        """
        Update states from input segment
        Additionlly update incremental states

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if self.tgt_lang is None and segment.tgt_lang is not None:
            self.tgt_lang = segment.tgt_lang
        if segment.is_empty:
            if segment.finished:
                self.target_finished = True
            return
        segment_content: UnitYTextDecoderOutput = segment.content
        content = segment_content.decoder_features
        token = segment_content.tokens
        self.source_indices = segment_content.target_indices
        self.source_token_list += token
        self.source = content


class NARUnitYUnitDecoderAgent(GenericAgent):  # type: ignore
    """Non-autoregressive unit decoder"""

    source_type = "text"
    target_type = "text"

    def __init__(
        self, model: UnitYNART2UModel, tokenizer: UnitTokenizer, args: Namespace
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.min_unit_chunk_size = args.min_unit_chunk_size
        self.d_factor = args.d_factor
        self.device = args.device
        self.dtype = args.dtype
        self.token_decoder = self.tokenizer.create_decoder()
        super().__init__(args)

    def build_states(self) -> NARUnitDecoderAgentStates:
        return NARUnitDecoderAgentStates()

    @property
    def max_len(self) -> int:
        return 200

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--min-unit-chunk-size",
            type=int,
            required=True,
            help="Minimal units to produce every chunk",
        )
        parser.add_argument(
            "--d-factor",
            type=float,
            default=1.0,
            help="scaling factor for duration prediction",
        )

    @torch.inference_mode()
    def policy(self, states: NARUnitDecoderAgentStates) -> Action:
        if states.target_finished:
            return WriteAction("", finished=True)

        if len(states.source_token_list) < 2:
            if not states.source_finished:
                return ReadAction()
            else:
                return WriteAction("", finished=True)

        print("[Debug] class NARUnitYUnitDecoderAgent :: func policy")
        # import pdb; pdb.set_trace()

        if control_switch.nARUnitYUnitDecoderAgent["save_flag"]:
            """
                功能: 存储模型的权重和输入、输出
            """
            msg = "[INFO]save nARUnitYUnitDecoderAgent"
            print(msg)
            
            copied_model = copy.deepcopy(self.model)
            copied_text_seqs = copy.deepcopy(states.source_indices)
            copied_text_decoder_output = copy.deepcopy(states.source)

            save_input_output_unit_decoder(
                model=copied_model,
                text_seqs=copied_text_seqs,
                text_decoder_output=copied_text_decoder_output,
                text_decoder_padding_mask=None,
                duration_factor=self.d_factor,
                film_cond_emb = None
            )

            # 获取环境变量 - 对应变量在执行脚本中设置
            weight_save_folder = control_switch.nARUnitYUnitDecoderAgent['weight_save_folder']
            linear_quantize_flag = control_switch.nARUnitYUnitDecoderAgent['quantize_flag']
            linear_quantize_bit = control_switch.nARUnitYUnitDecoderAgent['linear_quantize_bit']
            
            # 构建存储文件夹和存储名称
            weight_save_name = "decode_unit_weight"
            save_weight_of_decode_unit(copied_model.cpu(), weight_save_folder, weight_save_name, linear_quantize_flag, linear_quantize_bit)
            
            # del copied_seqs
            # del copied_model
            # gc.collect()     # 执行垃圾回收，以确保及时释放内存

            control_switch.nARUnitYUnitDecoderAgent["save_flag"] = False
        
        model_output, _, durations = self.model(
            text_decoder_output=states.source,
            text_decoder_padding_mask=None,
            text_seqs=states.source_indices,
            duration_factor=self.d_factor,
        )
        durations = durations[0]

        if states.source_finished and states.duration_start_index > 0:
            # We have to consider one more word for EOS, because we append an EOS at the end.
            if sum(durations[states.duration_start_index :]) == 0:
                # If you reach here, it means that the last source token is a silence token (e.g. punctuations)
                # In that case no need to consider one more token.
                return WriteAction("", finished=True)
            else:
                states.duration_start_index = max(states.duration_start_index - 1, 0)

        current_duration = sum(durations[states.duration_start_index :])

        if current_duration < self.min_unit_chunk_size:
            if not states.source_finished:
                # if current untranslated source result less than self.min_unit_chunk_size units
                return ReadAction()
            else:
                if current_duration == 0:
                    return WriteAction("", finished=True)

        unit_seqs = model_output.logits[0].argmax(dim=-1)
        index_start_offset = sum(durations[: states.duration_start_index])
        unit_seqs = unit_seqs[index_start_offset:].unsqueeze(0)
        units = self.token_decoder(unit_seqs)

        # minus one because we add a ending_token on each s2t output phrase
        states.duration_start_index = len(durations) - 1

        return WriteAction(
            TextSegment(
                content=units,
                finished=states.source_finished,
                tgt_lang=states.tgt_lang,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: Any) -> NARUnitYUnitDecoderAgent:
        unity_model: UnitYModel = kwargs.get("unity_model", None)
        unit_tokenizer: UnitTokenizer = kwargs.get("unit_tokenizer", None)
        assert unity_model.t2u_model is not None and isinstance(
            unity_model.t2u_model, UnitYNART2UModel
        )
        return cls(model=unity_model.t2u_model, tokenizer=unit_tokenizer, args=args)


def save_weight_of_decode_unit(model, weight_save_folder, weight_save_name, linear_quantize_flag, linear_quantize_bit):
    import os
    # from seamless_communication.src.tools.model_weight_save import save_model_state_dict, save_model_structure
    # from seamless_communication.src.tools.quantization import quantize_Agnent3_OfflineWav2VecBertEncoderAgent
    from src.tools.weight_save.model_weight_save import save_model_state_dict, save_model_structure
    from src.tools.quantization.quantization import quantize, QuantizedLinear

    # 提示信息
    print(">" * 12, "save weight of decode_unit", ">" * 12)
    # 量化权重
    if linear_quantize_flag == True:
        model = quantize(model, weight_bit_width=linear_quantize_bit)
        # model.encoder = quantize(model.encoder, weight_bit_width=linear_quantize_bit)
        # model.decoder_frontend = quantize(model.decoder_frontend, weight_bit_width=linear_quantize_bit)
        # model.decoder = quantize(model.decoder, weight_bit_width=linear_quantize_bit)
        # model.final_proj = QuantizedLinear(
        #     weight_bit_width=linear_quantize_bit,
        #     weight=model.final_proj.weight,
        #     bias=model.final_proj.bias,
        #     dtype=model.final_proj.weight.dtype,
        #     device=model.final_proj.weight.device
        # ) 

    if linear_quantize_flag == True:
        if linear_quantize_bit == 4:
            weight_save_name += "_int4"
        elif linear_quantize_bit == 8:
            weight_save_name += "_int8"
        else:
            # origin
            weight_save_name += "_fp16"
    else:
        # origin
        weight_save_name += "_fp16"
    # 存储权重
    save_model_state_dict(model, weight_save_folder, weight_save_name)
    # 存储模型结构
    struct_save_name = "decode_unit_structure"
    save_model_structure(model, weight_save_folder, struct_save_name)
    # 提示信息
    print("<" * 12, "save weight of decode_unit", "<" * 12)


def save_input_output_unit_decoder(model, text_seqs, text_decoder_output, text_decoder_padding_mask, duration_factor, film_cond_emb):
    from src.tools.weight_save.model_weight_save import save_tensor
    # from seamless_communication.src.tools.model_weight_save import save_tensor

    save_dir = "./model_weight/Agent3_nARUnitYUnitDecoderAgent_input_output"
    save_tensor(text_decoder_output.cpu(), tensor_name="unit_decoder_input", save_dir=save_dir)

    encoder_output, encoder_padding_mask = model.encode(
        text_decoder_output, text_decoder_padding_mask
    )
    
    if model.prosody_proj is not None and film_cond_emb is not None:
            import pdb;pdb.set_trace()
            encoder_output = encoder_output + model.prosody_proj(film_cond_emb)

    decoder_output, decoder_padding_mask, durations = model.decode(
        encoder_output,
        encoder_padding_mask,
        text_seqs,
        duration_factor,
        film_cond_emb,
    )

    logits = model.final_proj(decoder_output)

    save_tensor(logits.cpu(), tensor_name="unit_decoder_output", save_dir=save_dir)