from torch.utils.data import Dataset

class PseudoData(Dataset):
    def __init__(self, data_x, data_y):
        super().__init__()
        self.data_x = data_x.tolist()

        self.data_y = data_y
        if self.data_y is not None:
            self.data_y = data_y.tolist()

    def __len__(self):
        return len(self.data_y)
    
    def __getitem__(self, idx):
        sample = {}

        sample['x'] = self.data_x[idx]

        if self.data_y is not None:
            sample['y'] = self.data_y[idx]

        return sample