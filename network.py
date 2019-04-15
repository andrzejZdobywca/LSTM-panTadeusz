import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

TRAIN_ON_GPU = torch.cuda.is_available()
class Network(nn.Module):
    def __init__(self, tokens, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(len(tokens), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, len(tokens))
    
    def init_hidden(self):
        if(TRAIN_ON_GPU):
            return (torch.randn(1,1,self.hidden_dim).cuda(), torch.randn(1,1,self.hidden_dim).cuda())
        return (torch.randn(1,1,self.hidden_dim), torch.randn(1,1,self.hidden_dim))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, hidden


def train(net, data, epochs=5, seq_size=32, clip=5, lr=0.0003, val_freq=0.2):

    # set optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # convert text to array of integers
    data = utils.convert_to_int(data)

    # divide data into training and validation set
    val_idx = int((len(data)*(1-val_freq)))
    data, val_data = data[:val_idx], data[val_idx:]

    if(TRAIN_ON_GPU):
        net.cuda()

    for e in range(epochs):
        # initialize hidden state
        hidden = net.init_hidden()
        running_loss = 0.0

        for i in range(len(data)-seq_size):
            x, y = torch.Tensor(data[i:(i+seq_size)]).long(), torch.Tensor(data[(i+1):(i+seq_size+1)]).long()
            if(TRAIN_ON_GPU):
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

            if(i % 250 == 0):
                print("train_loss aftter {0} = {1:.7f}".format(i, running_loss.item()))
                running_loss = 0.0