class Tokenizer:
    def __init__(self, vocabs, use_padding = True, max_padding=64, pad_token='[PAD]', unk_token='[UNK]') :
        self.idx_to_token = vocabs
        self.token_to_idx = {token : idx for idx, token in enumerate(vocabs)}
        
        self.use_padding = use_padding
        self.max_padding = max_padding

        self.pad_token = pad_token
        self.unk_token = unk_token

        self.unk_token_idx = self.token_to_idx[self.unk_token]
        self.pad_token_idx = self.token_to_idx[self.pad_token]

    def __call__(self, x:str) :
        token_ids = []

        # 토큰화
        for token in x.split():
            if token in self.token_to_idx:
                id = self.token_to_idx[token]
            else:
                id = self.unk_token_idx
            token_ids.append(id)        
        
        #padding 넣기
        if self.use_padding:
            token_ids = token_ids[:self.max_padding]
            add_cnt = self.max_padding - len(token_ids)
            token_ids.extend([self.pad_token_idx] * add_cnt)
        
        return token_ids