from text import _clean_text, _symbol_to_id


def phoneme_to_sequence(text, cleaner_names):
    sequence = []

    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
        return clean_text  # list


def use_phoneme(text):
    sequence = []

    for symbol in text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence  # list
