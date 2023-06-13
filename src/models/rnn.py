"""RNN structure model for time series modeling

Sourced from 
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/time-series/Simple_RNN.ipynb
"""
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(
            self, 
            input_size, 
            output_size,
            hidden_dim, 
            n_layers,
            criterion, 
            optimizer: str="adam", 
            learning_rate: float=0.01
        ):
        super(RNN, self).__init__()
        # hidden dimension
        self.hidden_dim=hidden_dim
        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # create criterion
        self.criterion = criterion
        # create optimizer attribute
        if optimizer.lower()=="adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) 
        else:
            self.optimizer = None

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        
        # get output and hidden output for x input and previous hidden dimension
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  
        # get final output 
        output = self.fc(r_out)
        
        return output, hidden

    def train(self, data, n_steps, print_every):

        # initialize hidden state
        h_0 = None
        # fit to training data and convert training data to tensors
        for batch_i, step in enumerate(range(n_steps)):
            x = data.x
            y = data.y
            # convert data into tensors
            x_tensor = torch.Tensor(x).unsqueeze(0)
            y_tensor = torch.Tensor(y)
            # outputs from rnn 
            pred, hidden = self(x_tensor, hidden)
            # represent memory, detach hidden state from history
            # to avoid over-backprop
            hidden = hidden.data
            # calculate the loss
            loss = self.criterion(pred, y_tensor)

            # zero gradients and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print every option for loss display
            if batch_i%print_every == 0:        
                print('Loss: ', loss.item())


if __name__=="__main__":
    test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2, criterion=nn.MSELoss())
    print(test_rnn)
