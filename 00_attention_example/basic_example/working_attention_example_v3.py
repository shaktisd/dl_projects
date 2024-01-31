'''
This is a working version of simple self attention .
v3: Refactoring code for a separate model class
'''
import numpy as np
import random, string
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(42)

class Task(object):
    def __init__(self, max_len=10, vocab_size=3, batch_size=5):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.universe = string.ascii_uppercase[0:self.vocab_size]
        alphabet_list = list(self.universe)
        self.label_encoder = LabelEncoder()
        numeric_labels = self.label_encoder.fit_transform(alphabet_list)
        numeric_labels = numeric_labels.reshape(-1, 1)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.one_hot_encoder.fit(numeric_labels)
        self.batch_size = batch_size

    def next_batch(self):
        x_batch = []
        y_batch = []
        for b in range(0, self.batch_size):
            alphabets = []
            for i in range(self.max_len):
                alphabets.append(random.choice(self.universe))
                # alphabets.append(random.randint(1,self.vocab_size))
            count_alphabets = Counter(alphabets)
            m = dict(count_alphabets.most_common())
            #print('x', alphabets)
            numeric_labels = self.label_encoder.transform(alphabets)
            numeric_labels = numeric_labels.reshape(-1, 1)
            #print('x numeric', numeric_labels)
            x = self.one_hot_encoder.transform(numeric_labels)
            #print('x one hot', x)
            y = [m.get(k, 0)/self.max_len for k in self.universe]
            #print('y', y)
            x_batch.append(x)
            y_batch.append(y)
            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return torch.tensor(x_batch, dtype=torch.float),torch.tensor(y_batch, dtype=torch.float)


class AttentionModel(nn.Module):
    def __init__(self,embedding_dim,vocab_size,max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(self.vocab_size, self.embedding_dim, bias=False)
        self.key = nn.Linear(self.vocab_size, self.embedding_dim, bias=False)
        self.value = nn.Linear(self.vocab_size, self.embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.embedding_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, v)
        return weighted


class Model(nn.Module):
    def __init__(self, max_len, embedding_dim, vocab_size):
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.linear1 = nn.Linear(self.max_len * self.embedding_dim, self.max_len * self.embedding_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.max_len * self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, weighted):
        flat_weights = weighted.view(weighted.shape[0], -1)
        output = self.linear1(flat_weights)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output

def main():
    max_len = 10
    vocab_size = 5
    embedding_dim = 512
    batch_size = 1000
    # Training loop
    num_epochs = 1000

    task = Task(max_len = max_len, vocab_size=vocab_size, batch_size=batch_size)
    self_attention_model = AttentionModel(embedding_dim=embedding_dim, vocab_size=vocab_size, max_len=max_len)
    self_attention_model.to(device)
    model = Model(embedding_dim=embedding_dim, vocab_size=vocab_size, max_len=max_len)
    model.to(device)
    # Loss function and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self_attention_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        x, y = task.next_batch()
        x = x.to(device)
        y = y.to(device)

        model.train()
        self_attention_model.train()

        # Forward pass
        weighted = self_attention_model(x)
        y_pred = model(weighted)
        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            acc = accuracy_fn(y_true=y * max_len,
                              y_pred=torch.round(y_pred * max_len),
                              vocab_size=vocab_size)
            # Print training information
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Train Accuracy: {acc}')

def accuracy_fn(y_true, y_pred, vocab_size):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/(len(y_pred) * vocab_size)) * 100
    return acc

if __name__ == '__main__':
    main()