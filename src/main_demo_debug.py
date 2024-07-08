from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STJointVADAgent,
)
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

from config import LyngorBuildFlags, ModelSaveWeightFlags

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
)
from .main_demo import get_s2st_delayed_targets

print("building system from dir")

###### Flag
###### >>>>>> >>>>>> >>>>>>

# Set build flag by os env.
lyngor_build_flag = LyngorBuildFlags()
lyngor_build_flag.build_offlineWav2VecBertEncoderAgent = "False"
lyngor_build_flag.build_vocoderAgent = "False"
lyngor_build_flag.init_os_env()

# save weight flag by os env.
save_weight_flag = ModelSaveWeightFlags()
save_weight_flag.save_offlineWav2VecBertEncoderAgent = "True"
save_weight_flag.save_vocoderAgent = "False"
save_weight_flag.init_os_env()

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
    wav_file="./input/reading.wav",
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
    plt.savefig("./output/wave" + str(i) + ".png", dpi=300)

wave_total = []
for i in range(len(wave)):
    wave_total.extend(wave[i])
print(len(wave_total))
# save as wav
soundfile.write(
    "./output/reading.wav",
    wave_total,
    target_sample_rate,
    format="WAV",
    subtype="PCM_16",
)
