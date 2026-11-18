
def build_vocab(text):
    return sorted(set(text))


def encode(text, stoi):
    fallback = stoi.get(" ", 0)
    return [stoi.get(char, fallback) for char in text]


def decode(tokens, itos):
    return "".join(itos[int(token)] for token in tokens)
