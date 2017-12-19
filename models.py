import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import settings

class BaselineModel(nn.Module):

    def __init__(self, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=256, hidden_size=hidden_size, num_layers=1)
        self.output_layer = nn.Linear(hidden_size, 4)
        self.h0 = Variable(torch.randn(1, 1, hidden_size))

    def forward(self, sequence):
        output, hn = self.rnn(sequence, self.h0)
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "BaseLine_{}".format(self.hidden_size)


class SimpleLSTM(nn.Module):

    def __init__(self, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1)
        self.output_layer = nn.Linear(hidden_size, 4)
        self.h0 = Variable(torch.randn(1, 1, hidden_size))
        self.c0 = Variable(torch.randn(1, 1, hidden_size))

    def forward(self, sequence):
        output, hn = self.lstm(sequence, (self.h0, self.c0))
        predictions = self.output_layer(output)
        return predictions

    def get_name(self):
        return "SimpleLSTM_{}".format(self.hidden_size)

class ConvLSTM(nn.Module):
    
    def __init__(self, hidden_size=100, num_layers=1, kernel_size=1, kernel_nb=200, batch_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.kernel_nb =  kernel_nb
        self.batch_size =  batch_size
        #CNN layer
        self.conv1 = nn.Conv2d(1, kernel_nb ,(kernel_size,256))
        #LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1)
        self.hidden = self.init_hidden(num_layers, batch_size, hidden_size)

    def init_hidden(self, num_layers, batch_size, hidden_size):
        return (Variable(torch.zeros(1 * num_layers, batch_size, hidden_size)), Variable(torch.zeros(1 * num_layers, batch_size, hidden_size)))

    def forward(self, sequence):
        #CNN 
        cnn_x = torch.transpose(sequence, 0, 1)
        cnn_x = F.relu(self.conv1(cnn_x))
        #LSTM
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
    
        #CNN_LSTM
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_lstm_out = torch.cat((cnn_x,lstm_out),0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)
        return cnn_lstm_out


    def get_name(self):
        return "CNN and LSTM_{}".format(self.hidden_size, self.num_layers, self.kernel_size, self.kernel_nb, self.batch_size)





      
