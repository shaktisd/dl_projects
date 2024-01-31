from typing import List

import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class Dictionary(object):
    def __init__(self, file_path):
        self.index2word = dict()
        self.word2index = dict()
        self.OOV = 'OOV'
        self.PAD = ''
        if file_path is None:
            file_path = "small_imdb.csv"
        df = pd.read_csv(file_path)

        max_len = 32
        df['small_review'] = df['review'].str.split(n=max_len).str[:max_len].str.join(' ')
        df['small_review'] = df['small_review'].apply(lambda x:  re.sub(r'[^a-z ]+', '', x))
        text = " ".join(df['small_review'].str.lower().to_list())
        text = re.sub(r'[^a-z ]+', '', text)
        words = text.split()
        self.vocab = sorted(list(set(words)))
        self.vocab.insert(0,self.OOV)
        self.vocab.insert(1, self.PAD)
        for index, word in enumerate(self.vocab):
            self.index2word[index] = word
            self.word2index[word] = index

    def encode(self,text: str) -> List[int]:
        tokens = text.lower().split()
        encoded_str = [self.word2index.get(token,self.word2index[self.OOV]) for token in tokens]
        return encoded_str

    def decode(self, encoded_str: List[int]) -> str:
        decoded_str = [self.index2word[index] for index in encoded_str ]
        decoded_str = " ".join(decoded_str)
        return decoded_str


class AttentionModel(nn.Module):
    def __init__(self,embedding_dim,vocab_size,max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        dim_q = embedding_dim
        dim_k = embedding_dim
        dim_v = embedding_dim

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.query = nn.Linear(dim_q, self.embedding_dim, bias=False)
        self.key = nn.Linear(dim_k, self.embedding_dim, bias=False)
        self.value = nn.Linear(dim_v, self.embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding_layer(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.embedding_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, v)
        return weighted


def get_data(df: pd.DataFrame, data: Dictionary) -> (torch.Tensor, torch.Tensor):
    x = []
    y = []
    max_size = 32
    for index, row in df.iterrows():
        encoded = data.encode(row['small_review'])[0:max_size]
        length = len(encoded)
        encoded = np.pad(encoded, (0, max_size - length), 'constant')
        x.append(torch.tensor(encoded))

        y.append(row['sentiment'])
    x = np.vstack(x)
    y = np.vstack(y)

    return torch.tensor(x, dtype=torch.int), torch.tensor(y, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, max_len, embedding_dim, vocab_size):
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.self_attention_model = AttentionModel(embedding_dim=embedding_dim, vocab_size=vocab_size, max_len=max_len)
        self.linear1 = nn.Linear(self.max_len * self.embedding_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weighted = self.self_attention_model(x)
        flat_weights = weighted.view(weighted.shape[0], -1)
        output = self.linear1(flat_weights)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output

def main():
    embedding_dim = 16
    max_len = 32
    num_epochs = 300
    file_path = "imdb_clean.csv"
    data = Dictionary(file_path=file_path)
    df = pd.read_csv(file_path)

    df['small_review'] = df['review'].str.lower().str.split(n=max_len).str[:max_len].str.join(' ')
    df['small_review'] = df['small_review'].apply(lambda x: re.sub(r'[^a-z ]+', ' ', x))
    vocab_size = len(data.vocab)
    model = Model(embedding_dim=embedding_dim, vocab_size=vocab_size, max_len=max_len)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x, y = get_data(df, data)

    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        y_pred = model(x)
        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            acc = (y_pred.round() == y).float().mean()
            # Print training information
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Train Accuracy: {acc}')


if __name__ == '__main__':
    main()