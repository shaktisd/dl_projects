import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


if __name__ == '__main__':
    data_tensor = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    target_tensor = torch.tensor([0,1,0,0])

    custom_dataset = CustomDataset(data_tensor, target_tensor)
    batch_size = 2

    dataloader = DataLoader(custom_dataset, batch_size=batch_size,shuffle=True )

    for batch in dataloader:
        data, target = batch
        print('Data ', data)
        print('Target ', target)


