import random
import re
from pysle import isletool
from phonemizer import phonemize


CMU_IPA_CONSONANTS = [
    "b", "ʧ", "d", "ð", "f", "g", "h", "ʤ", "k", "l",
    "m", "n", "ŋ", "p", "r", "s", "ʃ", "t", "θ", "v",
    "w", "j", "z", "ʒ"
]

CMU_IPA_VOWELS = [
    "ɑ", "æ", "ə", "ʌ", "ɔ", "ɛ", "ɚ", "ɝ", "ɪ", "i",
    "ʊ", "u", "aʊ", "aɪ", "eɪ", "oʊ", "ɔɪ"
]

# stops and fricatives
CMU_IPA_FCD =  [
    "b", "g", "ʤ", "k", 
    "p", "s", "ʃ", "t", 
    "z", "ʒ", "θ",
]

def generate_missing(text):
    missing_type = random.choice([generate_phone_missing_syllable_deletion, generate_phone_missing_consonant_deletion])
    result = ""
    
    if missing_type == generate_phone_missing_consonant_deletion:
        con_del = generate_phone_missing_consonant_deletion(text, random.randint(1,  2))
        result = con_del[0] if con_del[1] else generate_phone_missing_syllable_deletion(text)[0]
    else:
        syl_del = generate_phone_missing_syllable_deletion(text)
        result = syl_del[0] if syl_del[1] else generate_phone_missing_consonant_deletion(text, random.randint(1,  2))[0]
    return result

def generate_phone_missing_syllable_deletion(text):  
    #print("entered syllable")
    dysfluency = False
    isle = isletool.Isle("ISLEdict.txt")
    words = text.split(" ")
    poly = False
    shuffled_indices = random.sample(range(len(words)), k=len(words))
    chosen_index = 0
    for index in shuffled_indices:
        chosen = words[index]
        chosen_index = index
        chosen = chosen.rstrip('.')
        syl_dictionary =  isle.lookup(chosen)[0].toDict()['syllabificationList']
        print(syl_dictionary)
        if len(syl_dictionary[0]) <= 1:
            continue
        else:
            poly = True
            break

    if poly: 
        unstressed_indices = [i for i, syllable in enumerate(syl_dictionary[0]) 
                              if i != (len(syl_dictionary[0]) - 1) 
                              and all("ˈ" not in phon for phon in syllable)]
        if unstressed_indices:
            drop = random.choice(unstressed_indices)
            del syl_dictionary[0][drop]
            generatedWord = ''.join(''.join(phon for phon in syl) for syl in syl_dictionary[0])
            firstH = ' '.join(words[:chosen_index])
            secondH = ' '.join(words[chosen_index + 1:])

            dysfluency = True
            generatedSentence = phonemize(firstH, language='en-us', backend='espeak', 
                                      strip=True, preserve_punctuation=True, with_stress=True) + "  " +  generatedWord + "  " + phonemize(secondH, language='en-us', backend='espeak', 
                                      strip=True, preserve_punctuation=True, with_stress=True)
        else:
            # no dysfluency generated
            generatedSentence  = phonemize(text, language='en-us', backend='espeak', 
                                      strip=True, preserve_punctuation=True, with_stress=True)
    return generatedSentence, dysfluency


def generate_phone_missing_consonant_deletion(text, delCount):
    #print("entered consonant")
    con_count = 0
    consonantDeleted = False
    phonemizedText = phonemize(text, language='en-us', backend='espeak', 
                        strip=True, preserve_punctuation=True, with_stress=True)
    words = phonemizedText.split(" ")
    deleted_letter =  ' '
    result = phonemizedText

    shuffled_indices = random.sample(range(len(words)), k=len(words))

    for index in shuffled_indices:
        chosen_word = words[index]
        if chosen_word[-1] not in CMU_IPA_FCD:
            # could also change this to CMU_IPA_CONSONANTS
            continue
        else:
            deleted_letter = chosen_word[-1]
            replacedWord = chosen_word[:(len(chosen_word)-1)]
            result = re.sub(re.escape(chosen_word), replacedWord, phonemizedText, count=1)
            consonantDeleted = True
            con_count += 1
            if (con_count == delCount):
                break

    return result, consonantDeleted  


result = generate_missing("Please call Stella")
print("result:{}".format(result))
