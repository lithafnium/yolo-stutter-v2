import torch
import random
from enum import Enum
from generate_utils import *
from models import SynthesizerTrn
import utils

class StutterType(Enum):
    PHONEBLOCK=0
    PHONEREP=1

def generate_phone_block(text, net_g, hps, sample_rate=22050, out_file="block.wav"):
    phonemes = get_phonemes(text)
    stn_tst = get_text(phonemes, hps)

    audio, durations = infer_audio(stn_tst, net_g)

    unit_duration = 256 / sample_rate

    durations = durations * unit_duration
    durations = durations.flatten()

    timestamps = get_time_transcription(durations, stn_tst)

    words = phonemes.split(" ")
    text_indexes, _ = get_indexes(words)

    chosen = random.sample(text_indexes, k=1)[0]
    chosen_timestamp = timestamps[chosen]

    audio = np.array(audio)
    print(len(audio))
    print(chosen_timestamp)
    insert_noise(audio, chosen_timestamp["start"] * sample_rate, silence_duration_sec=4, noise_std_dev=0.01, output_path=out_file)


def generate(text, net_g, hps, generate_type=StutterType.PHONEREP):
    if generate_type == StutterType.PHONEREP:
        for i in range(5):
            generate_phone_block(text, net_g, hps, out_file=f"block_{i}.wav")



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