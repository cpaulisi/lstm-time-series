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
            n_layers
        ):
        super(RNN, self).__init__()
        self = self.float()
        # hidden dimension
        self.hidden_dim=hidden_dim
        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # get output and hidden output for x input and previous hidden dimension
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size, seq_length, hidden_dim)
        r_out = r_out.view(batch_size, seq_length, self.hidden_dim)  
        # get final output 
        output = self.fc(r_out)
        
        return output, hidden

    def training_cycle(self, train_loader, val_loader, epochs, print_every, criterion, optimizer):
        # initialize the hidden state 
        for epoch in range(epochs):
            hidden=None
            # defining the training data 
            running_train_loss = 0
            self.train()
            for x, y in train_loader:
                # set training mode
                # convert data into Tensors
                x_tensor = torch.Tensor(x).float()
                y_tensor = torch.Tensor(y).unsqueeze(-1).float()
                # outputs from the rnn from train
                prediction, hidden = self(x_tensor, hidden)

                ## Representing Memory ##
                # make a new variable for hidden and detach the hidden state from its history
                # this way, we don't backpropagate through the entire history
                hidden = hidden.data
                # calculate the loss
                loss = criterion(prediction, y_tensor)
                # zero gradients
                optimizer.zero_grad()
                # perform backprop and update weights
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            

            # display loss and predictions
            if (epoch % print_every == 0) or (epoch==(epochs-1)): 
                # calculate final_train_loss
                train_loss = running_train_loss/len(train_loader)
                # set to eval mode and run validation
                self.eval()
                hidden=None
                running_val_loss = 0
                for x_test, y_test in val_loader:
                    x_tensor_test = torch.Tensor(x_test).float()
                    y_tensor_test = torch.Tensor(y_test).unsqueeze(-1).float()
                    # outputs from the rnn for validation
                    test_prediction, hidden = self(x_tensor_test, hidden)
                    ## Representing Memory ##
                    # make a new variable for hidden and detach the hidden state from its history
                    # this way, we don't backpropagate through the entire history
                    hidden = hidden.data
                    # calculate the loss
                    val_loss = criterion(test_prediction, y_tensor_test)
                    running_val_loss += val_loss

                val_loss = running_val_loss/len(val_loader)
                print("Epoch: %d, train loss: %1.5f, val loss: %1.5f" % (epoch, train_loss, val_loss))


if __name__=="__main__":
    test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)
    print(test_rnn)
