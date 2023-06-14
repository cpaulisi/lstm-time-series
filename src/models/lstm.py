"""This class instantiates an LSTM model

Author: Cullen Paulisick
"""

import torch
from torch import nn
import matplotlib.pyplot as plt


class TimeSeriesLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        n_layers,
        drop_prob: float = 0.5
    ):
        super(TimeSeriesLSTM, self).__init__()
        # declare parameters as float
        self = self.float()
        # hidden dimension
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # check if GPU is available
        self.train_on_gpu = torch.cuda.is_available()
        if (self.train_on_gpu):
            print('Training on GPU!')
        else:
            print(
                'No GPU available, training on CPU; consider making n_epochs very small.')
        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be
        # the batch_size

        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
            # TODO: define the LSTM
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        seq_length = x.size(1)

        # get output and hidden output for x input and previous hidden
        # dimension
        r_out, hidden = self.lstm(x, hidden)
        out = self.dropout(r_out)
        # shape output to be (batch_size, seq_length, hidden_dim)
        out = out.view(batch_size, seq_length, self.hidden_dim)
        # get final output
        output = self.fc(out)

        return output, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (
                weight.new(
                    self.n_layers,
                    batch_size,
                    self.hidden_dim).zero_().cuda(),
                weight.new(
                    self.n_layers,
                    batch_size,
                    self.hidden_dim).zero_().cuda())
        else:
            hidden = (
                weight.new(
                    self.n_layers,
                    batch_size,
                    self.hidden_dim).zero_(),
                weight.new(
                    self.n_layers,
                    batch_size,
                    self.hidden_dim).zero_())

        return hidden

    def training_cycle(
        self,
        batch_size,
        train_loader,
        val_loader,
        epochs,
        print_every,
        criterion,
        optimizer,
        writer=None,
        clip: int = 5
    ):
        """Cycle through a set number of epochs and retrieve output
        """
        # initialize the hidden state
        for epoch in range(epochs):
            # set to train
            self.train()
            # set to gpu if available
            if (self.train_on_gpu):
                self.cuda()
            # initialize hidden state
            hidden = self.init_hidden(batch_size=batch_size)
            # defining the training data
            running_train_loss = 0

            for x, y in train_loader:
                # set training mode
                # convert data into Tensors
                x_tensor = torch.Tensor(x).float()
                y_tensor = torch.Tensor(y).unsqueeze(-1).float()
                # transfer to device for gpu if available
                if (self.train_on_gpu):
                    x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()

                # zero gradients
                optimizer.zero_grad()
                # outputs from the rnn from train
                prediction, hidden = self(x_tensor, hidden)

                ## Representing Memory ##
                # make a new variable for hidden and detach the hidden state from its history
                # this way, we don't backpropagate through the entire history
                hidden = tuple([each.data for each in hidden])
                # calculate the loss
                loss = criterion(prediction, y_tensor)

                # perform backprop and update weights
                loss.backward()
                # use clip_grad_norm to safeguard against exploding gradient
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
                running_train_loss += loss.item()

            # calculate final train_loss
            train_loss = running_train_loss / len(train_loader)
            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)

            # run validation step at each epoch
            val_hidden = self.init_hidden(batch_size)
            # set to eval mode and run validation
            self.eval()
            running_val_loss = 0
            for x_test, y_test in val_loader:
                x_tensor_test = torch.Tensor(x_test).float()
                y_tensor_test = torch.Tensor(y_test).unsqueeze(-1).float()
                if (self.train_on_gpu):
                    x_tensor_test, y_tensor_test = x_tensor_test.cuda(), y_tensor_test.cuda()
                # outputs from the rnn for validation
                test_prediction, val_hidden = self(x_tensor_test, val_hidden)
                ## Representing Memory ##
                # make a new variable for hidden and detach the hidden state from its history
                # this way, we don't backpropagate through the entire history
                val_hidden = tuple([each.data for each in val_hidden])
                # val_hidden = val_hidden.data
                # calculate the loss
                val_loss = criterion(test_prediction, y_tensor_test)
                running_val_loss += val_loss

            self.train()
            val_loss = running_val_loss / len(val_loader)
            if writer:
                writer.add_scalar("Loss/val", val_loss, epoch)

            # display loss and predictions
            if (epoch % print_every == 0) or (epoch == (epochs - 1)):
                print(
                    "Epoch: %d, train loss: %1.5f, val loss: %1.5f" %
                    (epoch, train_loss, val_loss))

        # flush if writer being used
        if writer:
            writer.flush()

        return train_loss, val_loss

    def display_inference(self, x, y, display: str = "actual"):
        """Display an inference in matplot lib comparing actuals/input to predicted

        Toggle display to see prediction versus either input data or actual data.
        Displays first example in batch from loader.
        Displays input/actual in red, and prediction in blue
        """
        # set to eval
        self.eval()
        time_steps = range(x.shape[1])
        x_tensor_val = torch.Tensor(x[0, :, :]).unsqueeze(0).float()
        y_tensor_val = torch.Tensor(y[0, :]).unsqueeze(0).float()
        # y_tensor_val = torch.Tensor(val_y).unsqueeze(-1).float()
        hidden = self.init_hidden(batch_size=1)
        prediction, hidden = self(x_tensor_val, hidden)

        # toggle between displaying actuals and input
        if display.lower() == "actual":
            plt.plot(
                time_steps,
                y_tensor_val.data.numpy().flatten(),
                'r-')  # actual
        elif display.lower() == "input":
            plt.plot(
                time_steps,
                x_tensor_val.data.numpy()[
                    :,
                    :,
                    0].flatten(),
                'r-')  # input

        # plot prediction
        plt.plot(
            time_steps,
            prediction.data.numpy().flatten(),
            'b-')  # predictions
        plt.show()
