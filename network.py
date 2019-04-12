import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def read_text(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

def convert_to_int(data):
    chars = set(data)
    chars_as_int = dict([(x, i) for i, x in enumerate(chars)])
    data_as_int = [chars_as_int[x] for x in data]
    return data_as_int

# dodac one-hot embedding
# zrobic podzial na batche
# dodac pobieranie pliku zamiast uzywanie

class Network(nn.Module):
    def __init__(self, tokens):
        # dodac parametr embedding_dim
        self.embedding_dim = 64
        self.hidden_dim = 64

        super().__init__()
        self.chars = tokens
        self.chars_as_int = dict(enumerate(tokens))
        self.embedding = nn.Embedding(len(tokens), self.embedding_dim) # tutaj embedding_dim = 16

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, len(tokens))
    
    def init_hidden(self):
        self.hidden  = (torch.randn(1,1,self.hidden_dim), torch.randn(1,1,self.hidden_dim))
        return self.hidden

    def forward(self, x, hidden):
        
        x, hidden = self.lstm(x, hidden)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, hidden


def train(net, data):
    print(optim)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    embedding = net.embedding

    val_idx = int((len(data)*0.9))
    data, val_data = data[:val_idx], data[val_idx:]
    hidden = net.init_hidden()
    running_loss = 0.0

    seq_size = 64
    for i in range(len(data)-seq_size):
        x, y = torch.Tensor(data[i:(i+seq_size)]).long(), torch.Tensor(data[(i+1):(i+seq_size+1)]).long()
        x = embedding(x)
        x = x.view(seq_size, 1, -1)
        hidden = tuple([each.data for each in hidden])
        
        net.zero_grad()
        out, hidden = net(x, hidden)
        loss = criterion(out, y)
        running_loss += loss
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        if(i % seq_size == 0):
            print(running_loss.item())
            running_loss = 0.0
        # print(x, y)



data = read_text('./text.txt')
# data = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
tokens = set(data)
data_as_int = convert_to_int(data)
print(data_as_int[100:200])

net = Network(tokens)
train(net, data_as_int)