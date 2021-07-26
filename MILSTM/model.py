import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MILSTMTrainer(nn.Module):
    def __init__(self, conf):
        super(MILSTMTrainer, self).__init__()
        self.conf = conf

        self.model = MILSTM(conf['device'], conf['topK'], conf['input_size'], conf['hidden_size'], conf['num_layers'], conf['num_output'], conf['drop_out'])
        self.loss = nn.MSELoss(reduction='mean')
        self.Adam = optim.Adam(self.model.parameters(), lr=conf['learning_rate'])
        self.SGD = optim.SGD(self.model.parameters(), lr=conf['learning_rate'], momentum=0.9)

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


class MILSTM(nn.Module):
    def __init__(self, device, topK, input_size, hidden_size, num_layers, num_output, drop_out):
        super(MILSTM, self).__init__()
        self.topK = topK

        self.pre_model = nn.LSTM(input_size, hidden_size, num_layers, dropout = drop_out, batch_first = True)
        self.model = MILSTMChain(device, topK, input_size, hidden_size, num_layers)
        self.filter = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        output = []
        for i in range(self.topK + 1):
            out_i, _ = self.pre_model(x[:, i, :, :])
            output.append(out_i)

        Y = output[0]
        P = torch.cat(output[1:]).mean(dim=0).unsqueeze(0)

        output = self.model(Y, P)
        result = self.filter(output[:, -1, :])

        return result


class MILSTMChain(nn.Module):
    def __init__(self, device, topK, input_size, hidden_size, num_layers, bias = True):
        super(MILSTMChain, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.topK = topK

        self.cell = MILSTMCell(device, input_size, hidden_size, bias)

    def initHidden(self):
        hidden_pair = self.cell.initHidden()
        return hidden_pair

    def forward(self, input, input_p):
        hidden_pair = self.initHidden()
        output = []
        x_list = torch.unbind(input, dim=1)
        xp_list = torch.unbind(input_p, dim=1)

        for i in range(len(x_list)):
            hidden_pair = self.cell(x_list[i], xp_list[i], hidden_pair)
            output.append(hidden_pair[0])

        return torch.cat(output, 0).view(input.size(0), # batch
                                         input.size(1), # seq len
                                         input.size(2)) # hidden size


class MILSTMCell(nn.Module):
    def __init__(self, device, input_size, hidden_size, bias = True):
        super(MILSTMCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.w_ih_p = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))

        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.w_hh_p = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))

        self.w_a = nn.Parameter(torch.Tensor(1, hidden_size))

        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_ih_p = nn.Parameter(torch.Tensor(2 * hidden_size))

            self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh_p = nn.Parameter(torch.Tensor(2 * hidden_size))

            self.b_a = nn.Parameter(torch.Tensor(1))

        self.resetParameters()

    def initHidden(self):
        h = torch.zeros(1, self.hidden_size).to(self.device)
        c = torch.zeros(1, self.hidden_size).to(self.device)

        return (h, c)


    def resetParameters(self):
        param_count = 0

        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():

            param_count += 1

            weight.data.uniform_(-stdv, stdv)

        self.w_ih_p.data.uniform_(-0, 0)
        self.w_hh_p.data.uniform_(-0, 0)
        self.b_ih_p.data.uniform_(-0, 0)
        self.b_hh_p.data.uniform_(-0, 0)

        assert param_count == 5 + (5 if self.bias else 0), \
            'There are some unnoticed parameters in MILSTM cell(s). param_count: {}'.format(param_count)

    def forward(self, input, input_p, hidden):
        hx, cx = hidden

        gates = F.linear(input, self.w_ih, self.b_ih) + F.linear(hx, self.w_hh, self.b_hh)
        gates_p = F.linear(input_p, self.w_ih_p, self.b_ih_p) + F.linear(hx, self.w_hh_p, self.b_hh_p)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate_p, cellgate_p = gates_p.chunk(2, 1)

        ingate = torch.sigmoid(ingate)
        ingate_p = torch.sigmoid(ingate_p)

        forgetgate = torch.sigmoid(forgetgate)

        cellgate = torch.tanh(cellgate)
        cellgate_p = torch.tanh(cellgate_p)

        outgate = torch.sigmoid(outgate)

        l = ingate * cellgate
        l_p = ingate_p * cellgate_p

        u = torch.tanh(F.linear(l, self.w_a, self.b_a))
        u_p = torch.tanh(F.linear(l_p, self.w_a, self.b_a))

        #print 'size of u: {}'.format(u.size())
        #print 'size of u_p: {}'.format(u_p.size())

        U = torch.cat((u, u_p), 1)

        #print 'U: {}'.format(U)

        A = F.softmax(U, 1)

        #print 'A: {}'.format(A)

        A = A.unsqueeze(1)
        #print "Size of A: {}".format(A.size())

        l = l.unsqueeze(1)
        l_p = l_p.unsqueeze(1)
        #print "Size of l: {}".format(l.size())

        l_combined = torch.cat((l, l_p), 1)
        #print "Size of l_combined: {}".format(l_combined.size())

        L = torch.bmm(A, l_combined).squeeze(1)
        #print "Size of L: {}".format(L.size())

        cy = (forgetgate * cx) + L
        hy = outgate * torch.tanh(cy)

        return hy, cy