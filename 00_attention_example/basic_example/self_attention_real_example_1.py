import torch
import torch.nn as nn
import torch.optim as optim

# Sample data
# Assuming you have a dataset with sequences of word embeddings
# Each sequence has shape (sequence_length, embedding_dim)
# Here, we create a random dataset for illustration
torch.manual_seed(42)
sample_data = torch.randn((100, 10, 300))  # 100 sequences, each with length 10 and embedding dimension 300
print(sample_data)

# SelfAttention model
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.embedding_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


# Instantiate the model
embedding_dim = 300  # Assuming word embeddings have a dimensionality of 300
self_attention_model = SelfAttention(embedding_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(self_attention_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Forward pass
    outputs = self_attention_model(sample_data)

    # Compute loss
    loss = criterion(outputs, sample_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training information
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# After training, you can use the model for inference
# For example, given a new sequence of word embeddings x_new, you can do:
# result = self_attention_model(x_new)
