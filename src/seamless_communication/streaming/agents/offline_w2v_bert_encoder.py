# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from fairseq2.data import SequenceData
from fairseq2.data.data_pipeline import Collater
from fairseq2.data.text import TextTokenizer
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.unity.model import UnitYModel
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment
from seamless_communication.streaming.agents.common import (
    AgentStates,
    NoUpdateTargetMixin,
)


class MyUnitYModel_1Input(torch.nn.Module):
    """Tmp Model 1. to fit torch.jit.trace interface.
    1 input model
    """

    def __init__(self, unity_model):
        super().__init__()
        self.unity_model = unity_model.to(torch.device("cpu"))
        self.unity_model.speech_encoder_frontend.to(torch.device("cpu"))
        self.unity_model.speech_encoder.to(torch.device("cpu"))

    def forward(self, seqs):
        seqs.to(torch.device("cpu"))
        output = self.unity_model.encode_speech(seqs, None)
        # output 的含义 : seqs, padding_mask
        return output[0]


class MyUnitYModel_2Input(torch.nn.Module):
    """Tmp Model 2. to fit torch.jit.trace interface.
    2 input model
    """

    def __init__(self, unity_model):
        super().__init__()
        self.unity_model = unity_model.to(torch.device("cpu"))
        self.unity_model.speech_encoder_frontend.to(torch.device("cpu"))
        self.unity_model.speech_encoder.to(torch.device("cpu"))

    def forward(self, seqs, pad_mask):
        return self.unity_model.encode_speech(seqs, pad_mask)


def build_encode_speech(encode_speech_model, model_input_size):
    """编译encode_speech模型

    Args:
        model (_type_): _description_
    """
    print("[Debug] Build Model - class OfflineWav2VecBertEncoderAgent")
    import pdb

    pdb.set_trace()
    # *1. Use MyUnitYModel to fit torch.jit.trace interface.
    mymodel = None
    if len(model_input_size) == 1:
        mymodel = (
            MyUnitYModel_1Input(
                encode_speech_model,
            )
            .to(torch.device("cpu"))
            .eval()
            .float()
        )
    elif len(model_input_size) == 2:
        mymodel = (
            MyUnitYModel_2Input(
                encode_speech_model,
            )
            .to(torch.device("cpu"))
            .eval()
            .float()
        )
    else:
        raise ValueError("model_input_size must be 1 or 2.")

    # *2. DLmodel load
    import lyngor as lyn

    lyn.debug()
    data_type = "float32"  # TODO
    lyn_model = lyn.DLModel()
    model_type = "Pytorch"
    dict_inshape = {}
    dict_inshape.update({"data": model_input_size[0]})
    lyn_model.load(mymodel, model_type, inputs_dict=dict_inshape)

    # *3. DLmodel build
    target = "apu"
    # lyn_module = lyn.Builder(target=target, is_map=True, cpu_arch='x86', cc="g++")
    lyn_module = lyn.Builder(target=target)

    out_path = "./encode_speech_model"
    opt_level = 3
    lyn_module.build(
        lyn_model.mod, lyn_model.params, opt_level, out_path=out_path, build_mode="auto"
    )


# type: ignore
class OfflineWav2VecBertEncoderAgent(NoUpdateTargetMixin, SpeechToSpeechAgent):
    """
    Incremental encoding of an wav2vec encoder output
    It update the whole encoder states every time when there is a new incoming segment.
    """

    def __init__(
        self,
        unity_model: UnitYModel,
        w2v2_encoder_config: Wav2Vec2EncoderConfig,
        text_tokenizer: TextTokenizer,
        args: Namespace,
    ) -> None:
        super().__init__(args)
        self.model = unity_model
        self.w2v2_encoder_config = w2v2_encoder_config
        self.collate = Collater(
            pad_value=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )
        self.device = args.device
        self.dtype = args.dtype
        self.min_starting_wait = args.min_starting_wait_w2vbert
        # build self.model.encode_speech

    @property
    def min_input_length(self) -> int:
        return self.w2v2_encoder_config.fbank_stride

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--min-starting-wait-w2vbert",
            default=None,
            type=int,
            help="Min starting wait in w2vbert",
        )

    @torch.inference_mode()
    def policy(self, states: AgentStates) -> Action:
        """
        The policy for encoder is always write
        only if the input is too short
        """
        if (
            self.min_starting_wait is not None
            and len(states.source) < self.min_starting_wait
            and not states.source_finished
        ):
            return ReadAction()

        if len(states.source) < self.min_input_length:
            if states.source_finished:
                return WriteAction({}, finished=states.source_finished)
            else:
                return ReadAction()

        inputs = torch.stack(states.source).to(device=self.device, dtype=self.dtype)
        src: SequenceData = self.collate(inputs)

        seqs, padding_mask = get_seqs_and_padding_mask(src)
        # >>>>>> kaige add.
        import os

        env_name = "SAVE_OFFLINEWAV2VECBERTENCODERAGENT"
        save_offlineWav2VecBertEncoderAgent = os.environ.get(env_name)
        if save_offlineWav2VecBertEncoderAgent == ["Flase", "True"][1]:
            msg = "[INFO]save offlineWav2VecBertEncoderAgent"
            print(msg)
            save_weight_of_encode_speech(self.model)

        import os

        env_name = "BUILD_OFFLINEWAV2VECBERTENCODERAGENT"
        build_offlineWav2VecBertEncoderAgent = os.environ.get(env_name)
        if build_offlineWav2VecBertEncoderAgent == ["Flase", "True"][1]:
            print("[build_offlineWav2VecBertEncoderAgent]")
            import pdb

            pdb.set_trace()

            # build_encode_speech
            if padding_mask != None:
                build_encode_speech(self.model, [seqs.shape, padding_mask.shape])
                # build_encode_speech(self.model.speech_encoder_frontend, [seqs.shape, padding_mask.shape])
                # build_encode_speech(self.model.speech_encoder, [seqs.shape, padding_mask.shape])
            else:
                build_encode_speech(self.model, [seqs.shape])
                # build_encode_speech(self.model.encode_speech, [seqs.shape])

            """
            type(self.model) : <class 'seamless_communication.models.unity.model.UnitYModel'>
            """
            """
            seqs.shape: torch.Size([1, 222, 80])
            seqs.shape: torch.Size([1, 254, 80])
            seqs.shape: torch.Size([1, 286, 80])
            seqs.shape: torch.Size([1, 318, 80])
            seqs.shape: torch.Size([1, 350, 80])
            seqs.shape: torch.Size([1, 382, 80])
            seqs.shape: torch.Size([1, 414, 80])
            seqs.shape: torch.Size([1, 446, 80])
            seqs.shape: torch.Size([1, 478, 80])
            seqs.shape: torch.Size([1, 510, 80])
            seqs.shape: torch.Size([1, 542, 80])
            seqs.shape: torch.Size([1, 574, 80])
            seqs.shape: torch.Size([1, 606, 80])
            seqs.shape: torch.Size([1, 638, 80])
            seqs.shape: torch.Size([1, 670, 80])
            seqs.shape: torch.Size([1, 702, 80])
            seqs.shape: torch.Size([1, 734, 80])
            seqs.shape: torch.Size([1, 766, 80])

            padding_mask always is None
            """
            print("+" * 20, "3. OfflineWav2VecBertEncoderAgent")
            print("encode_speech - seqs.shape:", seqs.shape)
            if padding_mask is None:
                print("encode_speech - padding_mask: None")
            else:
                print("encode_speech - padding_mask: ", padding_mask.shape)
        # <<<<<<
        encoder_output, _ = self.model.encode_speech(
            seqs,
            padding_mask,
        )

        return WriteAction(
            SpeechSegment(
                content=encoder_output,
                tgt_lang=states.tgt_lang,
                finished=states.source_finished,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def from_args(
        cls, args: Namespace, **kwargs: Dict[str, Any]
    ) -> OfflineWav2VecBertEncoderAgent:
        unity_model = kwargs.get("unity_model", None)
        assert isinstance(unity_model, UnitYModel)
        unity_config = kwargs.get("unity_config", None)
        assert unity_config is not None
        text_tokenizer = kwargs.get("text_tokenizer", None)
        assert isinstance(text_tokenizer, TextTokenizer)
        return cls(unity_model, unity_config.w2v2_encoder_config, text_tokenizer, args)


def save_weight_of_encode_speech(model):
    """
        调试信息:
            type model : Class seamless_communication.models.unity.model.UnitYModel
            type (model.speech_encoder) : <class 'seamless_communication.models.unity.adaptor_block.UnitYEncoderAdaptor'>

            model.speech_encoder._modules.keys()
                odict_keys(['inner', 'inner_layer_norm', 'proj1', 'activation', 'proj2', 'adaptor_layers', 'layer_norm'])
            model.speech_encoder._modules['inner']._modules.keys()
                odict_keys(['layers', 'layer_norm'])
    """
    # from seamless_communication.src.model_weight_save import save_model_state_dict, save_model_structure
    from model_weight_save import save_model_state_dict, save_model_structure

    # 提示信息
    print(">" * 12, "save weight of encode_speech", ">" * 12)
    # 构建存储文件夹和存储名称
    weight_save_folder = "./model_weight/Agent3_OfflineWav2VecBertEncoderAgent"
    weight_save_name = "encode_speech_weight"
    # 存储权重
    save_model_state_dict(model.speech_encoder, weight_save_folder, weight_save_name)
    # 存储模型结构
    save_model_structure(model.speech_encoder, weight_save_folder, weight_save_name)
    # 提示信息
    print("<" * 12, "save weight of encode_speech", "<" * 12)

    print("> Agent3_OfflineWav2VecBertEncoderAgent speech encoder 模型的权重和结构存储完毕 <")
    import pdb; pdb.set_trace()


# # 使用gzip压缩二进制文件
# with gzip.open('model_weights.bin.gz', 'wb') as bin_file:
#     with open('model_weights.csv', 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['Key', 'Data Type', 'Shape'])

#         for key, tensor in state_dict.items():
#             # 将权重写入二进制文件
#             bin_file.write(tensor.numpy().tobytes())

#             # 写入CSV文件的信息
#             csv_writer.writerow([key, tensor.dtype, list(tensor.size())])
