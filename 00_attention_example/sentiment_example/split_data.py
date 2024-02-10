# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
#
# TEST_SIZE = 0.1
# BATCH_SIZE = 64
# SEED = 42
#
# # generate indices: instead of the actual data we pass in integers instead
# train_indices, test_indices, _, _ = train_test_split(
#     range(len(data)),
#     data.targets,
#     stratify=data.targets,
#     test_size=TEST_SIZE,
#     random_state=SEED
# )
#
# # generate subset based on indices
# train_split = Subset(data, train_indices)
# test_split = Subset(data, test_indices)
#
# # create batches
# train_batches = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
# test_batches = DataLoader(test_split, batch_size=BATCH_SIZE)

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data = data_tensor
        self.target = target_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return the data and target tensors for the given index
        return self.data[index], self.target[index]

# Example usage:
# Assuming you have data and target tensors, you can create an instance of the custom dataset
data_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
target_tensor = torch.tensor([0, 1, 0])

custom_dataset = CustomDataset(data_tensor, target_tensor)

# Now you can use a DataLoader to iterate over the dataset in batches
from torch.utils.data import DataLoader

batch_size = 2
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    # batch is a tuple containing data and target tensors
    data_batch, target_batch = batch
    print("Data Batch:", data_batch)
    print("Target Batch:", target_batch)
