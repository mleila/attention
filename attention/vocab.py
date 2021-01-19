from attention.constants import SOS_token, EOS_token, UNK_token, PAD_token


class Vocabulary:

    def __init__(self, sos=SOS_token, eos=EOS_token, unk=UNK_token, pad=PAD_token):
        self._token_to_index = {}
        self._tokens = []

        # register special tokens
        self.sos = sos
        self.eos = eos
        self.unk = unk
        self.pad = pad

        # add special tokens
        self.add_token(sos)
        self.add_token(eos)
        self.add_token(unk)
        self.add_token(pad)

    def add_token(self, token):
        if token in self._token_to_index:
            return
        self._token_to_index[token] = len(self._tokens)
        self._tokens.append(token)

    def lookup_token(self, token):
        if token in self._token_to_index:
            return self._token_to_index[token]
        return self._token_to_index[self.unk]

    def lookup_indes(self, index):
        return self._tokens[index]

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        return self._tokens[index]
