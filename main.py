import torch
import network
import utils

TRAIN_ON_GPU = torch.cuda.is_available()

if(TRAIN_ON_GPU):
    print('Training on GPU!')
else: 
    print('No GPU available :(')

data = utils.read_text('./text.txt')
tokens = set(data)

#endtest
net = network.Network(tokens, 64, 512)
network.train(net, data)