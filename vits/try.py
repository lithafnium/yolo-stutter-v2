from phonemizer import phonemize

text = "Please ...call stella"
phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
print(phonemes)  # str
