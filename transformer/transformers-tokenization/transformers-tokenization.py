from typing import List, Dict

class SimpleTokenizer:

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:

        words = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        for text in texts:
            for word in text.lower().split():
                if word not in words:
                    words.append(word)

        self.vocab_size = len(words)

        self.word_to_id = {w: i for i, w in enumerate(words)}
        self.id_to_word = {i: w for i, w in enumerate(words)}

    def encode(self, text: str) -> List[int]:

        ids = []

        for word in text.lower().split():
            ids.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))

        return ids

    def decode(self, ids: List[int]) -> str:

        words = []

        for i in ids:
            words.append(self.id_to_word.get(i, self.unk_token))

        return " ".join(words)