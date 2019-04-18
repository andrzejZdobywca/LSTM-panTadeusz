import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
# TODO:
# sprawdzic czy uda mi sie zamienic na self.hidden zamiast hidden

TRAIN_ON_GPU = torch.cuda.is_available()
class Network(nn.Module):
    def __init__(self, tokens, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.to_int_mapping = dict([(x, i) for i, x in enumerate(tokens)])
        self.to_char_mapping = dict([(i, x) for i, x in enumerate(tokens)])

        self.embedding = nn.Embedding(len(tokens), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(tokens))
    
    def init_hidden(self, batch_size):
        if(TRAIN_ON_GPU):
            return (torch.randn(1, batch_size, self.hidden_dim).cuda(), torch.randn(1, batch_size, self.hidden_dim).cuda())
        return (torch.randn(1, batch_size,self.hidden_dim), torch.randn(1, batch_size, self.hidden_dim))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.fc(x)
        return x, hidden

    def predict(self, char, hidden, top_k):
        x = [utils.convert_to_int(char)]
        x = torch.Tensor(x).long()
        if(TRAIN_ON_GPU):
            x = x.cuda()
        x = self.embedding(x)
        # hidden = tuple([each.data for each in hidden])

        out, hidden = self.forward(x, hidden)
        p = F.softmax(out, dim=1).data

        if(TRAIN_ON_GPU):
            p = p.cpu()
        _, char = p.topk(1)
        char_id = char.item()
        new_char = self.to_char_mapping[char_id]
        return new_char, hidden

    def sample(self, size, prime='The', top_k=None):

        if(TRAIN_ON_GPU):
            self.cuda()
        else:
            self.cpu()
        
        self.eval()

        chars = [ch for ch in prime]
        hidden = self.init_hidden(1)
        for ch in prime:
            char, hidden = self.predict(ch, hidden, top_k=top_k)
        chars.append(char)
        
        for ii in range(size):
            char, hidden = self.predict(chars[-1], hidden, top_k=top_k)
            chars.append(char)

        return ''.join(chars)


def train(net, data, epochs=10, batch_size=32, seq_length=50, clip=5, lr=0.0003, val_freq=0.2, print_every=40):

    net.train()
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
        hidden = net.init_hidden(batch_size)
        running_loss = 0.0
        counter = 0
        for x, y  in utils.get_batches(data, batch_size, seq_length):
            counter += 1
            if(TRAIN_ON_GPU):
                x, y = x.cuda(), y.cuda()

            x = net.embedding(x)

            hidden = tuple([each.data for each in hidden])

            net.zero_grad()
            out, hidden = net(x, hidden)
            y = y.view(batch_size * seq_length).long()
            loss = criterion(out, y)
            running_loss += loss

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

            if(counter % print_every == 0):
                print("epoch: ", e)
                print("train_loss after {0} batches = {1:.7f}".format(counter, running_loss.item()))
                print()
                running_loss = 0.0