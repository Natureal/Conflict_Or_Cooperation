import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import math


class LSTMTrainer(nn.Module):
    def __init__(self, conf):
        super(LSTMTrainer, self).__init__()
        self.conf = conf

        if conf['model'] == 'PytorchLSTM':
            print 'Use PytorchLSTM'
            self.model = PytorchLSTM(conf['device'], conf['input_size'], conf['hidden_size'], conf['num_layers'], conf['num_output'], conf['drop_out'])
        else:
            print 'Use LSTM'
            self.model = LSTM(conf['device'], conf['input_size'], conf['hidden_size'], conf['num_layers'], conf['num_output'])

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

# === LSTM in the libarary of pytorch ===
class PytorchLSTM(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, num_output, drop_out):
        super(PytorchLSTM, self).__init__()
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

# === LSTM implemented from scratch ===
class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.model = LSTMChain(device, input_size, hidden_size, num_layers).to(device)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_list = self.model.initHidden()
        input, output = self.model(x, hidden_list)

        #X = torch.cat((output[:, -1, :], input[:, :, 3]), dim = 1)
        X = output[:, -1, :]
        result = self.classifier(X)

        return result

class LSTMChain(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, bias=True):
        super(LSTMChain, self).__init__()
        self.device = device
        self.num_layers = num_layers

        cells = []
        cells.append(LSTMCell(device, input_size, hidden_size, bias))
        for i in range(1, num_layers):
            cells.append(LSTMCell(device, hidden_size, hidden_size, bias))

        self.cell_list = nn.ModuleList(cells)

    def initHidden(self):
        hidden_list = []
        for i in range(self.num_layers):
            hidden_list.append(self.cell_list[i].initHidden())
        return hidden_list

    def forward(self, input, hidden_list):
        # input size: batch_size, time_length, number of event codes
        current_input = input
        output = []
        next_hidden = []

        for layer in range(self.num_layers):

            hidden = hidden_list[layer]
            output_inner = []

            for x in torch.unbind(current_input, dim=1):

                hidden = self.cell_list[layer](x, hidden)

                output_inner.append(hidden[0])

            next_hidden.append(hidden)
            current_input = torch.cat(output_inner, 0).view(output_inner[0].size(0), #batch_size
                                                            current_input.size(1),
                                                            output_inner[0].size(1))

        return input, current_input


class LSTMCell(nn.Module):

    def __init__(self, device, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.w_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.b_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.resetParameters()

    def initHidden(self):
        h = torch.zeros(1, self.hidden_size).to(self.device)
        c = torch.zeros(1, self.hidden_size).to(self.device)
        return (h, c)

    def resetParameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):

        #print 'size of input: {}'.format(input.size())
        #print 'size of hidden: {}'.format(hidden[0].size())

        hx, cx = hidden

        gates = F.linear(input, self.w_ih, self.b_ih) + F.linear(hx, self.w_hh, self.b_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

