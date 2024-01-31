import numpy as np
import random, string
from collections import Counter, OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim


class Task(object):
    def __init__(self, max_len=10, vocab_size=3):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def next_batch(self, batchsize=5):
        x_batch = []
        y_batch = []
        for b in range(0, batchsize):
            alphabets = []
            for i in range(random.randint(1, self.max_len)):
                alphabets.append(random.choice(string.ascii_uppercase[0:self.vocab_size]))
                # alphabets.append(random.randint(1,self.vocab_size))
            count_alphabets = Counter(alphabets)
            sorted_count_alphabets = sorted(count_alphabets.items(), key=lambda pair: pair[0])
            x = [[a] for a in alphabets]
            y = [[k[1]] for k in sorted_count_alphabets]

            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch


class Task1(object):

    def __init__(self, max_len=10, vocab_size=3):
        super(Task1, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        assert self.vocab_size <= 26, "vocab_size needs to be <= 26 since we are using letters to prettify LOL"

    def next_batch(self, batchsize=100):
        x = np.eye(self.vocab_size + 1)[np.random.choice(np.arange(self.vocab_size + 1), [batchsize, self.max_len])]
        y = np.eye(self.max_len + 1)[np.sum(x, axis=1)[:, 1:].astype(np.int32)]
        return x, y

    def prettify(self, samples):
        samples = samples.reshape(-1, self.max_len, self.vocab_size + 1)
        idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
        dictionary = np.array(list(' ' + string.ascii_uppercase))
        return dictionary[idx]


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim) # Q
        self.key = nn.Linear(embedding_dim, embedding_dim) # K
        self.value = nn.Linear(embedding_dim, embedding_dim) # V
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.embedding_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


def main():
    vocab_size = 3
    max_len = 10

    task = Task1(max_len=max_len, vocab_size=vocab_size)
    x, y = task.next_batch(batchsize=1)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    print(x)
    print(y)

    # Instantiate the model
    embedding_dim = vocab_size+1  # Assuming word embeddings have a dimensionality of 10
    self_attention_model = SelfAttention(embedding_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self_attention_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        # Forward pass
        output = self_attention_model(x)

        # Compute loss
        loss = criterion(output, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


if __name__ == '__main__':
    main()