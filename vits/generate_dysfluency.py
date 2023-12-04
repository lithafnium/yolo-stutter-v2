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
    print(text_indexes)
    chosen = random.sample(range(len(text_indexes)), k=1)[0]
    phone = symbols[symbol_indexes[chosen]]
    text_index = text_indexes[chosen]
    
    pause = '.' * random.randint(2, 3) 
    repeated = f"{(phone + pause) * random.randint(2, 4)}"  #1-4 repetitions in the yolo stutter paper

    generated = phonemes[:text_index] + repeated + phonemes[text_index + 1:]    
    print(generated)
    stn_tst = get_text(generated, hps)

    audio, durations = infer_audio(stn_tst, net_g)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription(durations, stn_tst)
    # print(timestamps)
    merged_timestamps = []
    i = 0
    while i < len(timestamps):
        phoneme = timestamps[i]["phoneme"]
        if phoneme == phone and i + 1 < len(timestamps) and timestamps[i + 1]["phoneme"] == ".":
            start_time = timestamps[i]["start"]
            while i + 1 < len(timestamps) and timestamps[i + 1]["phoneme"] == ".":
                i += 1
            end_time = timestamps[i]["end"]
            merged_timestamps.append({
                "phoneme": phone,
                "start": start_time,
                "end": end_time,
                "type": "rep"
            })
        else:
            merged_timestamps.append(timestamps[i])
        i += 1


    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": merged_timestamps
    }]
    # merge same
    for entry in label:
        new_phonemes = []
        i = 0
        while i < len(entry['phonemes']):
            phoneme = entry['phonemes'][i]
            if phoneme['type'] == 'rep':
                start = phoneme['start']
                while i + 1 < len(entry['phonemes']) and entry['phonemes'][i + 1]['type'] == 'rep':
                    i += 1
                end = entry['phonemes'][i]['end']
                new_phonemes.append({'phoneme': phoneme['phoneme'], 'start': start, 'end': end, 'type': 'rep'})
            else:
                new_phonemes.append(phoneme)
            i += 1
        entry['phonemes'] = new_phonemes

    with open('rep.json', 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)




def generate_phone_block(text, net_g, hps, sample_rate=22050, out_file="block.wav"):
    phonemes = get_phonemes(text)
    stn_tst = get_text(phonemes, hps)

    audio, durations = infer_audio(stn_tst, net_g)

    unit_duration = 256 / sample_rate

    durations = durations * unit_duration
    durations = durations.flatten()

    timestamps = get_time_transcription(durations, stn_tst)
    # json file
    # Create the final structure
    label = [{
        "start": timestamps[0]["start"],
        "end": timestamps[-1]["end"],
        "phonemes": timestamps
    }]

    words = phonemes.split(" ")
    print(words)
    text_indexes, _ = get_first_indexes(words)
    print(text_indexes)

    chosen = random.sample(text_indexes, k=1)[0]  # index
    print(chosen)
    chosen_timestamp = timestamps[chosen]  ##
    print(chosen_timestamp)
    silence_duration = random.randint(1,3) ## ->
    
    audio = np.array(audio)
    audio = insert_noise(audio, chosen_timestamp["start"]*sample_rate, silence_duration_sec=silence_duration, noise_std_dev=0.01)
    
    block_phoneme = {
        "phoneme": "",
        "start": chosen_timestamp["start"],
        "end": chosen_timestamp["start"] + silence_duration/2,
        "type": "block"
    }
    timestamps.insert(chosen, block_phoneme)

    for i in range(chosen + 1, len(timestamps)):
        timestamps[i]["start"] += silence_duration/2
        timestamps[i]["end"] += silence_duration/2
        timestamps[i]["start"] = round(timestamps[i]["start"], 3)
        timestamps[i]["end"] = round(timestamps[i]["end"], 3)

    
    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]

    with open('block.json', 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)




def generate_phone_miss(text, net_g, hps, out_file="missing.wav"):
    original_phonems = get_phonemes(text)
    print(original_phonems)
    phonemes_miss = generate_missing(text)
    print(phonemes_miss)
    stn_tst = get_text(phonemes_miss, hps)
    audio, durations = infer_audio(stn_tst, net_g)
    unit_duration = 256 / 22050

    durations = durations * unit_duration
    durations = durations.flatten()
    timestamps = get_time_transcription(durations, stn_tst)

    # find missing phoenems
    missing_phoneme_index = []
    j = 0
    for i in range(len(original_phonems)):
        if j >= len(phonemes_miss) or original_phonems[i] != phonemes_miss[j]:
            missing_phoneme_index.append(i)
        else:
            j += 1
    missing_phoneme = missing_phoneme_index[0]
    print(missing_phoneme)
    print(original_phonems[missing_phoneme])

    for i, phoneme in enumerate(original_phonems):
        if i == missing_phoneme:
            print("find it:{}, {}".format(i, phoneme))
            missing_start_end = timestamps[i - 1]['end']
            timestamps.insert(i, {
                "phoneme": phoneme,
                "start": missing_start_end,
                "end": missing_start_end,
                "type": "missing"
            })
            break

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]

    with open('missing.json', 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)


def generate(text, net_g, hps, generate_type=StutterType.PHONEMISS):
    if generate_type == StutterType.PHONEBLOCK:
        generate_phone_block(text, net_g, hps, out_file=f"block.wav")
    if generate_type == StutterType.PHONEREP:
        generate_phone_rep(text, net_g, hps)
    if generate_type == StutterType.PHONEMISS:
        generate_phone_miss(text, net_g, hps)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        prog="GenerateDysfluency",
        description="Automatically generate dysfluencies from a given text"
    )

    parser.add_argument("--texts", type=str, help="Input file of texts to put in and generate")
    parser.add_argument("--text", type=str, help="Input text to generate dysfluency")
    parser.add_argument("--output_dir", type=str, help="Output directory to put generated sound files", required=True, default="./generated")

    parser.add_argument("--model_config", type=str, help="path to VITS model config", default="./configs/vctk_base.json")
    parser.add_argument("--model", type=str, help="path to model weights", default="./saved_models/pretrained_vctk.pth")
    args = parser.parse_args()

    if args.text is None and args.texts is None:
        raise Exception("One of --texts or --text needs to be passed in") 
    hps = utils.get_hparams_from_file(args.model_config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model, net_g, None)
    
    generate("Please call Stella", net_g, hps)
