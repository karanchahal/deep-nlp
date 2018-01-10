import torch.nn as nn
from torch.autograd import Variable
import torch

class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
    
    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output,hidden)
        
        return output,hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.hidden_size))
        return result

class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self,input,hidden):
        output = self.embedding(input).view(1,1,-1)
        for i in range(self.n_layers):
            output =F.relu(output)
            output, hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output,hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.hidden_size))
        return result


class AttentionDecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,n_layers=1,max_length=10):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size*2,self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size*2,self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size,self.hidden_size)
        self.out = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,input,hidden,encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax( self.attention( torch.cat((embedded[0], hidden[0]),1), dim=1 ))
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat( (embeddded[0],attention_applied[0]) ,1)
        output = self.attention_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output,hidden)
        
        output =F.log_softmax(self.out(output[0]), dim=1)
        return output,hidden,attention_weights
        
    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.hidden_size))
        return result

