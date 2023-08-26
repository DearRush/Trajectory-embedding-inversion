import torch
import torch.nn as nn


class StackingGRUCell(nn.Module):
    """
    Multi-layer GRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()#ModuleList是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)#将若干个张量在dim维度上连接,生成一个扩维的张量
        return output, hn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.linear = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h = [None, None, None]):
        """
        Input:
        input (batch, embedding_size): input embeddings
        h (num_layers, batch, hidden_size): input hidden state
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (batch, embedding_size)"
        output = []

        for i in range(20):
            o, h = self.rnn(input, h)
            o = self.dropout(o)
            input = self.linear(o)
            output.append(o)
        output = torch.stack(output)  
        return output, h