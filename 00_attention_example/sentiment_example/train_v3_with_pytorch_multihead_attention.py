from typing import List

import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split


class Dictionary(object):
    def __init__(self, file_path, max_len):
        self.index2word = dict()
        self.word2index = dict()
        self.OOV = 'OOV'
        self.PAD = ''
        self.max_len = max_len
        if file_path is None:
            file_path = "small_imdb.csv"
        df = pd.read_csv(file_path)

        df['small_review'] = df['review'].str.split(n=self.max_len).str[:self.max_len].str.join(' ')
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
    def __init__(self,embedding_dim,vocab_size,max_len,qkv_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.qkv_dim = qkv_dim

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.query = nn.Linear(self.embedding_dim, self.qkv_dim, bias=False)
        self.key = nn.Linear(self.embedding_dim, self.qkv_dim, bias=False)
        self.value = nn.Linear(self.embedding_dim, self.qkv_dim, bias=False)
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


def get_data(df: pd.DataFrame, data: Dictionary, max_len=32) -> (torch.Tensor, torch.Tensor):
    x = []
    y = []
    for index, row in df.iterrows():
        encoded = data.encode(row['small_review'])[0:max_len]
        length = len(encoded)
        encoded = np.pad(encoded, (0, max_len - length), 'constant')
        x.append(torch.tensor(encoded))

        y.append(row['sentiment'])
    x = np.vstack(x)
    y = np.vstack(y)

    return torch.tensor(x, dtype=torch.int), torch.tensor(y, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, max_len, embedding_dim, vocab_size, qkv_dim, hidden, num_heads):
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.qkv_dim = qkv_dim
        self.hidden = hidden
        self.num_heads = num_heads

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.query = nn.Linear(self.embedding_dim, self.qkv_dim, bias=False)
        self.key = nn.Linear(self.embedding_dim, self.qkv_dim, bias=False)
        self.value = nn.Linear(self.embedding_dim, self.qkv_dim, bias=False)


        #self.self_attention_model = AttentionModel(embedding_dim=embedding_dim, vocab_size=vocab_size, max_len=max_len, qkv_dim=self.qkv_dim)
        self.multihead_attn = nn.MultiheadAttention(self.embedding_dim, self.num_heads, batch_first=True)

        self.linear1 = nn.Linear(self.max_len * self.qkv_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.embedding_dim, self.hidden)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(self.hidden, 1)

    def forward(self, x):

        x_embedding = self.embedding_layer(x)
        q = self.query(x_embedding)
        k = self.key(x_embedding)
        v = self.value(x_embedding)

        attn_output, attn_output_weights = self.multihead_attn(q, k, v)

        weighted = attn_output
        flat_weights = weighted.reshape(weighted.shape[0], -1)
        output = self.linear1(flat_weights)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.linear3(output)
        output = self.sigmoid(output)
        return output

def main():
    embedding_dim = 64
    max_len = 32
    num_epochs = 50
    mini_batch_size = 32
    qkv_dim = 64
    hidden = 16
    num_heads = 1

    file_path = "imdb_clean.csv"
    data = Dictionary(file_path=file_path, max_len=max_len)
    df = pd.read_csv(file_path)

    df['small_review'] = df['review'].str.lower().str.split(n=max_len).str[:max_len].str.join(' ')
    df['small_review'] = df['small_review'].apply(lambda x: re.sub(r'[^a-z ]+', ' ', x))
    vocab_size = len(data.vocab)
    print(f'Vocab Size {vocab_size}')

    model = Model(embedding_dim=embedding_dim, vocab_size=vocab_size, max_len=max_len, qkv_dim=qkv_dim, hidden=hidden, num_heads=num_heads)
    criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss
    #criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x, y = get_data(df, data)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42)
    batch_start = torch.arange(0, len(x_train), mini_batch_size)

    # Hold the best model
    #best_acc = -np.inf  # init to negative infinity
    #best_weights = None

    for epoch in range(num_epochs):
        #print(f"Epoch {epoch}")
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                #print(start)
                # take a batch
                x_batch = x_train[start:start + mini_batch_size]
                y_batch = y_train[start:start + mini_batch_size]
                # Forward pass
                y_pred = model(x_batch)
                # Compute loss
                loss = criterion(y_pred, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )

            # evaluate accuracy at end of each epoch
            model.eval()
            y_test_pred = model(x_test)
            #test_acc = (y_test_pred.round() == y_test).float().mean()
            print(f"Epoch: {epoch} Train Loss: {loss} Train Acc: {acc} ")
            print_accuracy_threshold(y_test_pred, y_test)

    # Save Model
    print("Saving Model")
    torch.save(model.state_dict(), 'imdb_sentiment_model.pth')

def print_accuracy_threshold(y_test_pred, y_test):
    start = 0.0
    end = 1.0
    gap = 0.1
    for i in torch.arange(start, end + gap, gap):
        result = (torch.eq((y_test_pred - i > 0.0).float(), y_test) ).float().mean()
        print(i, result)

if __name__ == '__main__':
    main()