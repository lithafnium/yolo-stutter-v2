CMU_IPA_MAPPING = {
    'B': "b", 'CH': "ʧ", 'D': "d", 'DH': "ð", 'F': "f", 'G': "g",
    'HH': "h", 'JH': "ʤ", 'K': "k", 'L': "l", 'M': "m", 'N': "n",
    'NG': "ŋ", 'P': "p", 'R': "r", 'S': "s", 'SH': "ʃ", 'T': "t",
    'TH': "θ", 'V': "v", 'W': "w", 'Y': "j", 'Z': "z", 'ZH': "ʒ",
    'AA0': "ɑ", 'AA1': "ɑ", 'AA2': "ɑ", 'AE0': "æ", 'AE1': "æ", 'AE2': "æ",
    'AH0': "ə", 'AH1': "ʌ", 'AH2': "ʌ", 'AO0': "ɔ", 'AO1': "ɔ", 'AO2': "ɔ",
    'EH0': "ɛ", 'EH1': "ɛ", 'EH2': "ɛ", 'ER0': "ɚ", 'ER1': "ɝ", 'ER2': "ɝ",
    'IH0': "ɪ", 'IH1': "ɪ", 'IH2': "ɪ", 'IY0': "i", 'IY1': "i", 'IY2': "i",
    'UH0': "ʊ", 'UH1': "ʊ", 'UH2': "ʊ", 'UW0': "u", 'UW1': "u", 'UW2': "u",
    'AW0': "aʊ", 'AW1': "aʊ", 'AW2': "aʊ", 'AY0': "aɪ", 'AY1': "aɪ", 'AY2': "aɪ",
    'EY0': "eɪ", 'EY1': "eɪ", 'EY2': "eɪ", 'OW0': "oʊ", 'OW1': "oʊ", 'OW2': "oʊ",
    'OY0': "ɔɪ", 'OY1': "ɔɪ", 'OY2': "ɔɪ"
}


def cmu_to_ipa(cmu_string):

    cmu_symbols = cmu_string.split()
    ipa_result = [CMU_IPA_MAPPING.get(symbol, '..') if symbol != 'sp' else '..' for symbol in cmu_symbols]

    return ' '.join(ipa_result)


input_string = "AE1 S K HH ER0 T AH0 B R IH1 NG DH IY1 Z TH IH1 NG Z W IH0 DH HH ER0 F ER0 M DH AH0 S T AO1 R"
output_string = cmu_to_ipa(input_string)


print(output_string)
