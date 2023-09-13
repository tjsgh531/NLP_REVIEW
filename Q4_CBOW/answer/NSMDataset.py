from torch.utils.data import Dataset

class NSMDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        super().__init__()

        self.data_df = data_df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx) :
        sample_raw = self.data_df.iloc[idx]
        sample = {}

        sample['doc'] = str(sample_raw["document"])

        sample['label'] = int(sample_raw["label"])
        assert sample['label'] in set([0, 1])

        if self.tokenizer is not None:
            sample['doc_ids'] = self.tokenizer(sample['doc'])

        return sample 
