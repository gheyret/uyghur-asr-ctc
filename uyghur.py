import re

class Uyghur():
    def __init__(self, ):
        self.uyghur_latin = "abcdefghijklmnopqrstuvwxyz éöü’" 
        self._vocab_list = [self.pad_char, self.sos_char,self.eos_char] + list(self.uyghur_latin) # $ for padding char. index must be 0
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def encode(self, s):
        s = s.replace("-", ' ').replace(",", ' ').replace(".", ' ').replace("!", ' ').replace("?", ' ').replace("'","’")
        s = re.sub('\s+',' ',s).strip().lower()
        seq = [self.vocab_to_idx(v) for v in s if v in self.uyghur_latin]
        return seq

    def decode(self, seq):
        vocabs = []
        for idx in seq:
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or  idx == self.eos_idx:
                break
            elif idx == self.sos_idx:
                pass
            else:
                vocabs.append(v)
        s = re.sub('\s+',' ',"".join(vocabs)).strip()
        return s

    def vocab_to_idx(self, vocab):
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

    def vocab_list(self):
        return self._vocab_list

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def pad_idx(self):
        return self.vocab_to_idx(self.pad_char)

    @property
    def sos_idx(self):
        return self.vocab_to_idx(self.sos_char)

    @property
    def eos_idx(self):
        return self.vocab_to_idx(self.eos_char)

    @property
    def pad_char(self):
        return "<pad>"

    @property
    def sos_char(self):
        return "<sos>"

    @property
    def eos_char(self):
        return "<eos>"


uyghur_latin = Uyghur()
