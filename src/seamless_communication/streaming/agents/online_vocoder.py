# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from seamless_communication.models.vocoder.loader import load_vocoder_model
from seamless_communication.streaming.agents.common import AgentStates
from simuleval.agents import TextToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment

from config.config import ControlSwitch
control_switch = ControlSwitch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class VocoderAgent(TextToSpeechAgent):  # type: ignore
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        logger.info(
            f"Loading the Vocoder model: {args.vocoder_name} on device={args.device}, dtype={args.dtype}"
        )
        self.vocoder = load_vocoder_model(
            args.vocoder_name, device=args.device, dtype=args.dtype
        )
        self.vocoder.eval()

        self.sample_rate = args.sample_rate
        self.tgt_lang = args.tgt_lang
        self.speaker_id = args.vocoder_speaker_id

    @torch.inference_mode()
    def policy(self, states: AgentStates) -> WriteAction:
        """
        The policy is always write if there are units
        """
        units = states.source

        if len(units) == 0 or len(units[0]) == 0:
            if states.source_finished:
                return WriteAction([], finished=True)
            else:
                return ReadAction()

        tgt_lang = states.tgt_lang if states.tgt_lang else self.tgt_lang
        u = units[0][0]

        # >>>>>>
        import os
        if control_switch.VocoderAgent["build_flag"]:
            print("[Debug] class NARUnitYUnitDecoderAgent :: func policy")
            show_vocoder_info(self.vocoder, u, tgt_lang, self.speaker_id)
            # import pdb; pdb.set_trace()
            build_vocoder_agent(self.vocoder, u, tgt_lang, self.speaker_id)
        # <<<<<<

        wav = self.vocoder(u, tgt_lang, self.speaker_id, dur_prediction=False)
        states.source = []

        return WriteAction(
            SpeechSegment(
                content=wav[0][0].tolist(),
                finished=states.source_finished,
                sample_rate=self.sample_rate,
                tgt_lang=tgt_lang,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--vocoder-name",
            type=str,
            help="Vocoder name.",
            default="vocoder_v2",
        )
        parser.add_argument(
            "--vocoder-speaker-id",
            type=int,
            required=False,
            default=-1,
            help="Vocoder speaker id",
        )

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: Dict[str, Any]) -> VocoderAgent:
        return cls(args)


# >>>>>>
def show_vocoder_info(vocoder_model, u, tgt_lang, speaker_id):
    # import pdb; pdb.set_trace()
    msg = "="*30 + "show_vocoder_info func" + "="*30 + "\n"
    msg += f"type(vocoder_model) : {type(vocoder_model)}" + "\n"
    msg += f"type(u) : {type(u)}" + "\n"
    msg += f"u.shape : {u.shape}" + "\n"
    msg += f"u.dtype : {u.dtype}" + "\n"
    msg += f"type(tgt_lang) : {type(tgt_lang)}" + "\n"
    msg += f"tgt_lang : {tgt_lang}" + "\n"
    msg += f"type(speaker_id) : {type(speaker_id)}" + "\n"
    msg += f"speaker_id : {speaker_id}" + "\n"
    print(msg)


class VocoderWrapper(torch.nn.Module):
    """
        To Fit torch.jit.trace()
        Make sure input is tensor, output is tensor.
    """

    def __init__(
        self,
        vocoder_model,  # vocoder model
        lang_list,  # vocoder forward 2 input
        spkr_list=None,  # vocoder forward 3 input
        dur_prediction: bool = True  # vocoder forward 4 input
    ):
        super().__init__()
        self.vocoder_model = vocoder_model
        self.code_generator = vocoder_model.code_generator  # vocoder init 1
        self.lang_spkr_idx_map = vocoder_model.lang_spkr_idx_map  # vocoder init 2
        self.lang_list = lang_list
        self.spkr_list = spkr_list
        self.dur_prediction = dur_prediction

    def forward(
        self,
        units: torch.Tensor  # torch.Tensor
    ) -> torch.Tensor:
        return self.vocoder_model(units, self.lang_list, self.spkr_list, self.dur_prediction)


def build_vocoder_agent(vocoder_model,
                        u,  # vocoder forward 1 input
                        tgt_lang,  # vocoder forward 2 input
                        vocoder_speaker_id,  # vocoder forward 3 input
                        dur_prediction=True  # vocoder forward 4 input
                        ):
    print("[Debug] Build Model - class VocoderAgent")
    import pdb; pdb.set_trace()
    # *1. Use MyUnitYModel to fit torch.jit.trace interface.
    mymodel = VocoderWrapper(vocoder_model, tgt_lang,
                           vocoder_speaker_id, dur_prediction)
    mymodel = mymodel.to("cpu").float()

    # *2. DLmodel load
    import lyngor as lyn
    lyn.debug()
    data_type = 'long'
    lyn_model = lyn.DLModel()
    model_type = 'Pytorch'
    dict_inshape = {}
    dict_inshape.update({'data':[u.shape, data_type]})
    lyn_model.load(mymodel, model_type, inputs_dict = dict_inshape)

    # *3. DLmodel build
    target = "apu"
    # lyn_module = lyn.Builder(target=target, is_map=True, cpu_arch='x86', cc="g++")
    lyn_module = lyn.Builder(target=target)

    out_path = "./encode_speech_model"
    opt_level = 3
    lyn_module.build(lyn_model.mod, lyn_model.params, opt_level, out_path=out_path, build_mode="auto")


# <<<<<<
