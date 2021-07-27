from torch.utils.data import Dataset
import pickle




class ExpDataset(Dataset):
    def __init__(self, input_file):

        with open(input_file, "rb") as file:
            self.table = pickle.load(file)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        return self.table[idx]