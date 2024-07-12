from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STJointVADAgent,
)
import os

def insert_python_path():
    import os
    import sys

    # 获取当前文件的绝对路径
    current_path = os.path.abspath(__file__)
    # 获取上级目录（父目录）
    parent_path = os.path.dirname(current_path)
    # 获取上上级目录（祖父目录）
    grandparent_path = os.path.dirname(parent_path)

    # 将上级目录和上上级目录添加到Python搜索路径中
    # sys.path.insert0, parent_path)
    sys.path.insert(0, grandparent_path)

    # 打印sys.path以确认添加成功
    # print(sys.path)

insert_python_path()

import io
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import mmap
import numpy
import soundfile
import torchaudio
import torch

from collections import defaultdict

# from IPython.display import Audio, display
from pathlib import Path
from pydub import AudioSegment

from seamless_communication.inference import Translator
from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover


import math
from simuleval.data.segments import SpeechSegment, EmptySegment
from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STVADAgent,
)

from simuleval.utils.arguments import cli_argument_list
from simuleval import options


from typing import Union, List
from simuleval.data.segments import Segment, TextSegment
from simuleval.agents.pipeline import TreeAgentPipeline
from simuleval.agents.states import AgentStates

from config.config import LyngorBuildFlags, ModelSaveWeightFlags

# sample_rate : 是什么含义？
SAMPLE_RATE = 16000

from main_demo import AudioFrontEnd, OutputSegments
from main_demo import (
    get_audiosegment,
    reset_states,
    get_states_root,
    plot_s2st,
    build_streaming_system,
    run_streaming_inference,
    get_s2st_delayed_targets,
)

print("building system from dir")

###### Flag
###### >>>>>> >>>>>> >>>>>>

### Set build flag by os env.

from config.config import ControlSwitch
control_switch = ControlSwitch()

control_switch.offlineWav2VecBertEncoderAgent.update({
    'save_flag': False,
    'weight_save_folder': "./datas/model/Agent3_OfflineWav2VecBertEncoderAgent_weight",
    'quantize_flag': True,
    'linear_quantize_bit': 4,
    "build_flag": False,
})

control_switch.unitYMMATextDecoderAgent.update({
    'save_flag': False,
    'weight_save_folder': "./datas/model/Agent4_UnitYMMATextDecoderAgent_weight",
    'quantize_flag': True,
    'linear_quantize_bit': 4
})

control_switch.nARUnitYUnitDecoderAgent = {
    'save_flag': True,
    'weight_save_folder': "./datas/model/Agent6_nARUnitYUnitDecoderAgent_weight",
    'quantize_flag': False,
    'linear_quantize_bit': 4
}

###### <<<<<< <<<<<< <<<<<<

agent_class = SeamlessStreamingS2STJointVADAgent
tgt_lang = "spa"

model_configs = dict(
    source_segment_size=320,
    device="cuda:0",
    dtype="fp16",
    min_starting_wait_w2vbert=192,
    decision_threshold=0.5,
    min_unit_chunk_size=50,
    no_early_stop=True,
    max_len_a=0,
    max_len_b=100,
    task="s2st",
    tgt_lang=tgt_lang,
    block_ngrams=True,
    detokenize_only=True,
    # unity_model_name = "/home/fengkaige/codespace/seamless/seamless-streaming-card/",
    # unity_model_name="/home/fengkaige/codespace/seamless/seamless-streaming-card/seamless_streaming_unity.yaml",
)
system = build_streaming_system(model_configs, agent_class)
print("finished building system")

# import pdb; pdb.set_trace()


source_segment_size = 320  # milliseconds
audio_frontend = AudioFrontEnd(
    wav_file="./datas/input/reading.wav",
    segment_size=source_segment_size,
)

system_states = system.build_states()

# you can pass tgt_lang at inference time to change the output lang.
# SeamlessStreaming supports 36 speech output languages, see https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md#supported-languages
# in the Target column for `Sp` outputs.
delays, prediction_lists, speech_durations, target_sample_rate = (
    run_streaming_inference(system, audio_frontend, system_states, tgt_lang)
)


# target_samples, intervals = get_s2st_delayed_targets(delays, target_sample_rate, prediction_lists, speech_durations)

# plot_s2st("./input/reading.wav", target_samples, target_sample_rate, intervals, delays, prediction_lists)

print(delays)
print(prediction_lists.keys())
# print(speech_durations)
# print(target_sample_rate)

print("text outputs ", prediction_lists["s2tt"])
wave = prediction_lists["s2st"]
for i in range(len(wave)):
    print("wave", i)
    print("length", len(wave[i]))
    print("duration", len(wave[i]) / target_sample_rate)
    plt.plot(wave[i])
    plt.savefig("./datas/output/wave" + str(i) + ".png", dpi=300)

wave_total = []
for i in range(len(wave)):
    wave_total.extend(wave[i])
print(len(wave_total))
# save as wav
soundfile.write(
    "./datas/output/reading.wav",
    wave_total,
    target_sample_rate,
    format="WAV",
    subtype="PCM_16",
)
