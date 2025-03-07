from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return 0


    def __getitem__(self, idx):
        return None, None
