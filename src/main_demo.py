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
    SeamlessStreamingS2STVADAgent,)

from simuleval.utils.arguments import cli_argument_list
from simuleval import options


from typing import Union, List
from simuleval.data.segments import Segment, TextSegment
from simuleval.agents.pipeline import TreeAgentPipeline
from simuleval.agents.states import AgentStates

from config import LyngorBuildFlags, ModelSaveWeightFlags

# sample_rate : 是什么含义？
SAMPLE_RATE = 16000

class AudioFrontEnd:
    def __init__(self, wav_file, segment_size) -> None:
        self.samples, self.sample_rate = soundfile.read(wav_file)
        # print(self.sample_rate, SAMPLE_RATE)
        print('sample_rate', self.sample_rate)
        assert self.sample_rate == SAMPLE_RATE
        # print(len(self.samples), self.samples[:100])
        self.samples = self.samples  # .tolist()
        self.segment_size = segment_size
        self.step = 0
        print('len samples', len(self.samples))

    def send_segment(self):
        """
        This is the front-end logic in simuleval instance.py
        """

        num_samples = math.ceil(self.segment_size / 1000 * self.sample_rate)
        print('num_samples of this segment', num_samples)

        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                samples = self.samples[self.step:]
                is_finished = True
            else:
                samples = self.samples[self.step: self.step + num_samples]
                is_finished = False
            self.step = min(self.step + num_samples, len(self.samples))

            segment = SpeechSegment(
                content=samples,
                sample_rate=self.sample_rate,
                finished=is_finished,
            )
        else:
            # Finish reading this audio
            segment = EmptySegment(
                finished=True,
            )
        return segment


class OutputSegments:
    def __init__(self, segments: Union[List[Segment], Segment]):
        if isinstance(segments, Segment):
            segments = [segments]
        self.segments: List[Segment] = [s for s in segments]

    @property
    def is_empty(self):
        return all(segment.is_empty for segment in self.segments)

    @property
    def finished(self):
        return all(segment.finished for segment in self.segments)


def get_audiosegment(samples, sr):
    b = io.BytesIO()
    soundfile.write(b, samples, samplerate=sr, format="wav")
    b.seek(0)
    return AudioSegment.from_file(b)


def reset_states(system, states):
    if isinstance(system, TreeAgentPipeline):
        states_iter = states.values()
    else:
        states_iter = states
    for state in states_iter:
        state.reset()


def get_states_root(system, states) -> AgentStates:
    if isinstance(system, TreeAgentPipeline):
        # self.states is a dict
        return states[system.source_module]
    else:
        # self.states is a list
        return system.states[0]


def plot_s2st(source_file, target_samples, target_fs, intervals, delays, prediction_lists):
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = False

    source_samples, source_fs = soundfile.read(source_file)

    _, axes = plt.subplots(5, sharex=True, figsize=(25, 5))
    for ax in axes:
        ax.set_yticks([])

    axes[0].plot(
        numpy.linspace(0, len(source_samples) /
                       source_fs, len(source_samples)),
        source_samples,
    )

    axes[1].plot(
        numpy.linspace(0, len(target_samples) /
                       target_fs, len(target_samples)),
        target_samples,
    )

    start = 0
    for seg_index in range(len(intervals)):
        start, duration = intervals[seg_index]
        offset = delays["s2st"][seg_index]

        samples = target_samples[
            int((start) / 1000 * target_fs): int(
                (start + duration) / 1000 * target_fs
            )
        ]

        # Uncomment this if you want to see the segments without speech playback delay
        axes[2].plot(
            offset / 1000 +
            numpy.linspace(0, len(samples) / target_fs, len(samples)),
            -seg_index * 0.05 + numpy.array(samples),
        )
        axes[4].plot(
            start / 1000 +
            numpy.linspace(0, len(samples) / target_fs, len(samples)),
            numpy.array(samples),
        )

    from pydub import AudioSegment
    print("Output translation (without input)")
    # display(Audio(target_samples, rate=target_fs))
    print("Output translation (overlay with input)")
    source_seg = get_audiosegment(
        source_samples, source_fs) + AudioSegment.silent(duration=3000)
    target_seg = get_audiosegment(target_samples, target_fs)
    output_seg = source_seg.overlay(target_seg)
    # display(output_seg)

    delay_token = defaultdict(list)
    d = delays["s2tt"][0]
    for token, delay in zip(prediction_lists["s2tt"], delays["s2tt"]):
        if delay != d:
            d = delay
        delay_token[d].append(token)
    for key, value in delay_token.items():
        axes[3].text(
            key / 1000, 0.2, " ".join(value), rotation=40
        )


def build_streaming_system(model_configs, agent_class):
    parser = options.general_parser()
    parser.add_argument(
        "-f", "--f", help="a dummy argument to fool ipython", default="1")

    agent_class.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(model_configs))
    system = agent_class.from_args(args)
    return system


def run_streaming_inference(system, audio_frontend, system_states, tgt_lang):
    # NOTE: Here for visualization, we calculate delays offset from audio
    # *BEFORE* VAD segmentation.
    # In contrast for SimulEval evaluation, we assume audios are pre-segmented,
    # and Average Lagging, End Offset metrics are based on those pre-segmented audios.
    # Thus, delays here are *NOT* comparable to SimulEval per-segment delays
    delays = {"s2st": [], "s2tt": []}
    prediction_lists = {"s2st": [], "s2tt": []}
    speech_durations = []
    curr_delay = 0
    target_sample_rate = None

    while_count = 0
    while True:
        print("while_count:", while_count)
        while_count += 1

        input_segment = audio_frontend.send_segment()
        input_segment.tgt_lang = tgt_lang
        curr_delay += len(input_segment.content) / SAMPLE_RATE * 1000
        if input_segment.finished:
            # a hack, we expect a real stream to end with silence
            get_states_root(system, system_states).source_finished = True
        # Translation happens here

        # import pdb;pdb.set_trace()

        # output = system.pushpop(input_segment, system_states)

        output_segments = OutputSegments(
            system.pushpop(input_segment, system_states))
        if not output_segments.is_empty:
            for segment in output_segments.segments:
                # NOTE: another difference from SimulEval evaluation -
                # delays are accumulated per-token
                if isinstance(segment, SpeechSegment):
                    print("SpeechSegment")

                    pred_duration = 1000 * \
                        len(segment.content) / segment.sample_rate
                    speech_durations.append(pred_duration)
                    delays["s2st"].append(curr_delay)
                    prediction_lists["s2st"].append(segment.content)
                    target_sample_rate = segment.sample_rate
                elif isinstance(segment, TextSegment):
                    print("TextSegment")

                    delays["s2tt"].append(curr_delay)
                    prediction_lists["s2tt"].append(segment.content)
                    print(curr_delay, segment.content)
                elif isinstance(segment, EmptySegment):
                    # do nothing
                    print("EmptySegment")

        if output_segments.finished:
            print("End of VAD segment")
            reset_states(system, system_states)
        if input_segment.finished:
            # an assumption of SimulEval agents -
            # once source_finished=True, generate until output translation is finished
            assert output_segments.finished
            break
    return delays, prediction_lists, speech_durations, target_sample_rate


def get_s2st_delayed_targets(delays, target_sample_rate, prediction_lists, speech_durations):
    # get calculate intervals + durations for s2st
    intervals = []

    start = prev_end = prediction_offset = delays["s2st"][0]
    target_samples = [0.0] * int(target_sample_rate * prediction_offset / 1000)

    for i, delay in enumerate(delays["s2st"]):
        start = max(prev_end, delay)

        if start > prev_end:
            # Wait source speech, add discontinuity with silence
            target_samples += [0.0] * int(
                target_sample_rate * (start - prev_end) / 1000
            )

        target_samples += prediction_lists["s2st"][i]
        duration = speech_durations[i]
        prev_end = start + duration
        intervals.append([start, duration])
    return target_samples, intervals

if __name__ == "__main__":


    print("building system from dir")

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
    delays, prediction_lists, speech_durations, target_sample_rate = run_streaming_inference(
        system, audio_frontend, system_states, tgt_lang
    )


    # target_samples, intervals = get_s2st_delayed_targets(delays, target_sample_rate, prediction_lists, speech_durations)

    # plot_s2st("./input/reading.wav", target_samples, target_sample_rate, intervals, delays, prediction_lists)

    print(delays)
    print(prediction_lists.keys())
    # print(speech_durations)
    # print(target_sample_rate)

    print('text outputs ', prediction_lists["s2tt"])
    wave = prediction_lists["s2st"]
    for i in range(len(wave)):
        print('wave', i)
        print('length', len(wave[i]))
        print('duration', len(wave[i])/target_sample_rate)
        plt.plot(wave[i])
        plt.savefig("./output/wave"+str(i)+".png", dpi=300)

    wave_total = []
    for i in range(len(wave)):
        wave_total.extend(wave[i])
    print(len(wave_total))
    # save as wav
    soundfile.write("./output/reading.wav", wave_total,
                    target_sample_rate, format="WAV", subtype="PCM_16")
