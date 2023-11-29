import torch
import random
import re

from text import symbols
from enum import Enum
from generate_utils import *
from models import SynthesizerTrn
import utils
from process_miss import generate_missing

class StutterType(Enum):
    PHONEBLOCK=0
    PHONEREP=1
    PHONEMISS=2

def generate_phone_rep(text, net_g, hps, out_file="phonerep.wav"):
    phonemes = get_phonemes(text)
    words =  phonemes.split(" ")

    text_indexes, symbol_indexes = get_first_indexes(words)
    chosen = random.sample(range(len(text_indexes)), k=1)[0]
    phone = symbols[symbol_indexes[chosen]]
    text_index = text_indexes[chosen]
    
    pause = '.' * random.randint(2, 4) 
    repeated = f"{(phone + pause) * random.randint(2, 4)}"  #1-4 repetitions in the yolo stutter paper

    generated = phonemes[:text_index] + repeated + phonemes[text_index + 1:]    
    print(generated)
    stn_tst = get_text(generated, hps)

    audio, _ = infer_audio(stn_tst, net_g)

    write_audio_from_np(audio, out_file)

def generate_phone_block(text, net_g, hps, sample_rate=22050, out_file="block.wav"):
    phonemes = get_phonemes(text)
    stn_tst = get_text(phonemes, hps)

    audio, durations = infer_audio(stn_tst, net_g)

    unit_duration = 256 / sample_rate

    durations = durations * unit_duration
    durations = durations.flatten()

    timestamps = get_time_transcription(durations, stn_tst)

    words = phonemes.split(" ")
    text_indexes, _ = get_first_indexes(words)

    chosen = random.sample(text_indexes, k=1)[0]
    chosen_timestamp = timestamps[chosen]

    audio = np.array(audio)
    print(len(audio))
    print(chosen_timestamp)
    audio = insert_noise(audio, chosen_timestamp["start"] * sample_rate, silence_duration_sec=4, noise_std_dev=0.01)
    write_audio_from_np(audio, out_file)

def generate_phone_miss(text, net_g, hps, out_file="missing.wav"):
    phonemes_miss = generate_missing(text)
    print(phonemes_miss)
    stn_tst = get_text(phonemes_miss, hps)
    audio, _ = infer_audio(stn_tst, net_g)
    write_audio_from_np(audio, out_file)


def generate(text, net_g, hps, generate_type=StutterType.PHONEMISS):
    if generate_type == StutterType.PHONEBLOCK:
        generate_phone_block(text, net_g, hps, out_file=f"block.wav")
    if generate_type == StutterType.PHONEREP:
        generate_phone_rep(text, net_g, hps)
    if generate_type == StutterType.PHONEMISS:
        generate_phone_miss(text, net_g, hps)


if __name__ == "__main__": 

    hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    _ = utils.load_checkpoint("./saved_models/pretrained_vctk.pth", net_g, None)
    
    generate("Please call Stella", net_g, hps)
