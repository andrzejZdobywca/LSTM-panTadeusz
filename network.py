import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# setting training to gpu if avaiable
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


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
# dodac pobieranie pliku zamiast uzywanie sciezki

class Network(nn.Module):
    def __init__(self, tokens, embedding_dim, hidden_dim):
        # dodac parametr embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        super().__init__()
        self.embedding = nn.Embedding(len(tokens), embedding_dim) # tutaj embedding_dim = 16

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, len(tokens))
    
    def init_hidden(self):
        if(train_on_gpu):
            return (torch.randn(1,1,self.hidden_dim).cuda(), torch.randn(1,1,self.hidden_dim).cuda())
        return (torch.randn(1,1,self.hidden_dim), torch.randn(1,1,self.hidden_dim))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, hidden


def train(net, data, epochs=5, seq_size=64, clip=5, lr=0.0003, val_freq=0.2):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int((len(data)*(1-val_freq)))
    data, val_data = data[:val_idx], data[val_idx:]

    if(train_on_gpu):
        net.cuda()

    for e in range(epochs):
        hidden = net.init_hidden()
        running_loss = 0.0
        for i in range(len(data)-seq_size):
            x, y = torch.Tensor(data[i:(i+seq_size)]).long(), torch.Tensor(data[(i+1):(i+seq_size+1)]).long()
            if(train_on_gpu):
                x, y = x.cuda(), y.cuda()
            x = net.embedding(x)
            x = x.view(seq_size, 1, -1)
            hidden = tuple([each.data for each in hidden])
            net.zero_grad()
            out, hidden = net(x, hidden)
            loss = criterion(out, y)
            running_loss += loss
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            # print(seq_size)
            if(i % 250 == 0):
                print("train_loss aftter {0} = {1:.7f}".format(i, running_loss.item()))
                running_loss = 0.0
            # print(x, y)



data = read_text('./text.txt')
tokens = set(data)
data_as_int = convert_to_int(data)

net = Network(tokens, 64, 64)
train(net, data_as_int)