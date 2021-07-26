import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import math


class ELSTMTrainer(nn.Module):
    def __init__(self, conf):
        super(ELSTMTrainer, self).__init__()
        self.conf = conf

        if conf['model'] == 'ELSTM':
            print 'Use ELSTM'
            self.model = ELSTM(conf['device'], conf['input_size'], conf['hidden_size'], conf['num_layers'], conf['num_output'], conf['drop_out'])

        self.loss = nn.MSELoss(reduction='mean')
        self.Adam = optim.Adam(self.model.parameters(), lr = conf['learning_rate'])
        self.SGD = optim.SGD(self.model.parameters(), lr = conf['learning_rate'], momentum = 0.9)

    def update(self, data_train, data_val):
        self.model.train() # switch to training mode
        result = self.model(data_train)
        loss_val = self.loss(result, data_val)

        self.Adam.zero_grad()
        loss_val.backward()
        self.Adam.step()

        #self.SGD.zero_grad()
        #loss_val.backward()
        #self.SGD.step()

        return loss_val

    def eval(self, data_test, data_val):
        self.model.eval() # switch to evaluating mode
        result = self.model(data_test)

        #print 'result size: {}'.format(result.size())
        #print 'data_val: {}'.format(data_val.size())

        loss_val = self.loss(result, data_val)

        return loss_val, result

class ELSTM(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, num_output, drop_out):
        super(ELSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.model = nn.LSTM(input_size, hidden_size, num_layers, dropout = drop_out, batch_first = True) # (batch, sequence, feature)
        self.filter = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        # output of Pytorch LSTM: output, (h_n, c_n)
        # output: [batch, seq_len, num_directions * hidden_size], hidden state from the last layer
        # h_n: [num_layers * num_directions, batch, hidden_size], hidden state for t = seq_len
        # c_n: [num_layers * num_directions, batch, hidden_size], cell state for t = seq_len
        output, _ = self.model(x)

        result = self.filter(output[:, -1, :])

        return result
