#model
# File to return the Deep VO model.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from utils.util import correlate

# DeepVO model
class VINet(nn.Module):

    def __init__(self, imageWidth, imageHeight, activation='relu', parameterization='default', batchnorm=False, \
                 dropout=0.0, flownet_weights_path=None, numLSTMCells=1, hidden_units_LSTM=None, \
                 numFC=2, FC_dims=None):

        super(VINet, self).__init__()

        # Check if input image width and height are feasible
        self.imageWidth = int(imageWidth)
        self.imageHeight = int(imageHeight)
        if self.imageWidth < 64 or self.imageHeight < 64:
            raise ValueError('The width and height for an input image must be at least 64 px.')

        # Compute the size of the LSTM input feature vector.
        # There are 6 conv stages (some stages have >1 conv layers), which effectively reduce an
        # image to 1/64 th of its initial dimensions. Further, the final conv layer has 1024
        # filters, hence, numConcatFeatures = 1024 * (wd/64) * (ht/64) = (wd * ht) / 4
        self.numConcatFeatures = int((self.imageWidth * self.imageHeight) / 4)

        # Activation functions to be used in the network
        self.activation = activation

        # Whether or not batchnorm is required
        self.batchnorm = batchnorm

        # Whether or not dropout is required
        if dropout <= 0.0:
            self.dropout = False
        else:
            # Specify the drop_ratio
            self.dropout = True
            self.drop_ratio = dropout

        self.numLSTMCells = numLSTMCells
        self.hidden_units_LSTM = hidden_units_LSTM

        # Path to FlowNet weights
        if flownet_weights_path is not None:
            self.use_flownet = True
            self.flownet_weights_path = flownet_weights_path
        else:
            self.use_flownet = False

        """
        Initialize variables required for the network
        """

        # If we're using batchnorm, do not use bias for the conv layers
        self.bias = not self.batchnorm

        self.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=self.bias)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2, bias=self.bias)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2, bias=self.bias)
        self.conv_redir = nn.Conv2d(256, 32, kernel_size=1, stride=1)

        self.conv3_1 = nn.Conv2d(473, 256, bias=self.bias)
        self.conv4 = nn.Conv2d(256, 512, stride=2, bias=self.bias)
        self.conv4_1 = nn.Conv2d(512, 512, bias=self.bias)
        self.conv5 = nn.Conv2d(512, 512, stride=2, bias=self.bias)
        self.conv5_1 = nn.Conv2d(512, 512, bias=self.bias)
        self.conv6 = nn.Conv2d(512, 1024, stride=2, bias=self.bias)
        self.conv6_1 = nn.conv2d(1024, 1024)

        if self.numLSTMCells == 1:
            self.lstm1 = nn.LSTMCell(self.numConcatFeatures, self.hidden_units_LSTM[0])
            self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
            self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
        else:
            self.lstm1 = nn.LSTMCell(self.numConcatFeatures, self.hidden_units_LSTM[0])
            self.lstm2 = nn.LSTMCell(self.hidden_units_LSTM[0], self.hidden_units_LSTM[1])
            self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
            self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
            self.h2 = torch.zeros(1, self.hidden_units_LSTM[1])
            self.c2 = torch.zeros(1, self.hidden_units_LSTM[1])

        self.fc1 = nn.Linear(self.hidden_units_LSTM[self.numLSTMCells - 1], 128)

        # self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 32)


        self.fc_out = nn.Linear(32, 6)

    def forward(self, x, imu, reset_hidden=False):

        if not self.batchnorm:
            x1 = x[:,0:1,:,:]
            x2 = x[:,1:,:,:]

            x1 = (F.leaky_relu(self.conv1(x1)))
            x1 = (F.leaky_relu(self.conv2(x1)))
            x1 = (F.leaky_relu(self.conv3(x1)))

            x2 = (F.leaky_relu(self.conv1(x2)))
            x2 = (F.leaky_relu(self.conv2(x2)))
            x2 = (F.leaky_relu(self.conv3(x2)))

            redir = (F.leaky_relu(self.conv_redir(x1)))
            cor = correlate(x1,x2)
            x = torch.cat([redir, cor], dim=1)

            x = (F.leaky_relu(self.conv3_1(x)))
            x = (F.leaky_relu(self.conv4(x)))
            x = (F.leaky_relu(self.conv4_1(x)))
            x = (F.leaky_relu(self.conv5(x)))
            x = (F.leaky_relu(self.conv5_1(x)))
            x = (F.leaky_relu(self.conv(x)))
            x = (F.leaky_relu(self.conv6(x)))
            x = (self.conv6_1(x))


            if reset_hidden is True:
                if self.numLSTMCells == 1:
                    self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
                    self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
                else:
                    self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
                    self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
                    self.h2 = torch.zeros(1, self.hidden_units_LSTM[1])
                    self.c2 = torch.zeros(1, self.hidden_units_LSTM[1])

            if self.numLSTMCells == 1:
                self.h1, self.c1 = self.lstm1(imu, (self.h1, self.c1))
            else:
                self.h1, self.c1 = self.lstm1(imu, (self.h1, self.c1))
                self.h2, self.c2 = self.lstm2(self.h1, (self.h2, self.c2))

            # Forward pass through the FC layers
            if self.activation == 'relu':
                """
                output_fc1 = F.relu(self.fc1(lstm_final_output))
                """
                if self.numLSTMCells == 1:
                    output_fc1 = F.relu(self.fc1(self.h1))
                else:
                    output_fc1 = F.relu(self.fc1(self.h2))
                # output_fc1 = F.relu(self.fc1(self.h2))
                if self.dropout is True:
                    output_fc2 = F.dropout(F.relu(self.fc2(output_fc1)), p=self.drop_ratio, \
                                           training=self.training)
                else:
                    output_fc2 = F.relu(self.fc2(output_fc1))
            elif self.activation == 'selu':
                """
                output_fc1 = F.selu(self.fc1(lstm_final_output))
                """
                if self.numLSTMCells == 1:
                    output_fc1 = F.selu(self.fc1(self.h1))
                else:
                    output_fc1 = F.selu(self.fc1(self.h2))
                # output_fc1 = F.selu(self.fc1(self.h2))
                if self.dropout is True:
                    output_fc2 = F.dropout(F.selu(self.fc2(output_fc1)), p=self.drop_ratio, \
                                           training=self.training)
                else:
                    output_fc2 = F.selu(self.fc2(output_fc1))


            output = self.fc_out(output_fc2)


            return output

    # Initialize the weights of the network
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('# Linear')
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                # print('$ Conv2d')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.LSTMCell):
                # print('% LSTMCell')
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.)
                        bias = getattr(m, name)
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        bias.data[start:end].fill_(10.)

    # Detach LSTM hidden state (i.e., output) and cellstate variables to free up the
    # computation graph. Gradients will NOT flow backward through the timestep where a
    # detach is performed.
    def detach_LSTM_hidden(self):
        if self.numLSTMCells == 1:
            self.h1 = self.h1.detach()
            self.c1 = self.c1.detach()
        else:
            self.h1 = self.h1.detach()
            self.c1 = self.c1.detach()
            self.h2 = self.h2.detach()
            self.c2 = self.c2.detach()

    def reset_LSTM_hidden(self):
        if self.numLSTMCells == 1:
            self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
            self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
        else:
            self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
            self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
            self.h2 = torch.zeros(1, self.hidden_units_LSTM[1])
            self.c2 = torch.zeros(1, self.hidden_units_LSTM[1])

    # def load_flownet_weights(self):
    #
    #     if self.use_flownet is True:
    #
    #         flownet = torch.load(self.flownet_weights_path)
    #         cnn = flownet['state_dict']
    #
    #         if self.batchnorm is False:
    #
    #             cnn.conv1.weight.data = weights["conv1.0.weight"]
    #             cnn.conv1.bias.data = weights["conv1.0.bias"]
    #
    #             cnn.conv2.weight.data = weights["conv2.0.weight"]
    #             cnn.conv2.bias.data = weights["conv2.0.bias"]
    #
    #             cnn.conv3.weight.data = weights["conv3.0.weight"]
    #             cnn.conv3.bias.data = weights["conv3.0.bias"]
    #
    #             cnn.conv3_1.weight.data = weights["conv3_1.0.weight"]
    #             cnn.conv3_1.bias.data = weights["conv3_1.0.bias"]
    #
    #             cnn.conv4.weight.data = weights["conv4.0.weight"]
    #             cnn.conv4.bias.data = weights["conv4.0.bias"]
    #
    #             cnn.conv4_1.weight.data = weights["conv4_1.0.weight"]
    #             cnn.conv4_1.bias.data = weights["conv4_1.0.bias"]
    #
    #             cnn.conv5.weight.data = weights["conv5.0.weight"]
    #             cnn.conv5.bias.data = weights["conv5.0.bias"]
    #
    #             cnn.conv5_1.weight.data = weights["conv5_1.0.weight"]
    #             cnn.conv5_1.bias.data = weights["conv5_1.0.bias"]
    #
    #             cnn.conv6.weight.data = weights["conv6.0.weight"]
    #             cnn.conv6.bias.data = weights["conv6.0.bias"]
    #
    #         else:
    #             cnn.conv1.weight.data = weights["conv1.0.weight"]
    #             cnn.conv1_bn.weight.data = weights["conv1.1.weight"]
    #             cnn.conv1_bn.bias.data = weights["conv1.1.bias"]
    #             cnn.conv1_bn.running_mean.data = weights["conv1.1.running_mean"]
    #             cnn.conv1_bn.running_var.data = weights["conv1.1.running_var"]
    #
    #             cnn.conv2.weight.data = weights["conv2.0.weight"]
    #             cnn.conv2_bn.weight.data = weights["conv2.1.weight"]
    #             cnn.conv2_bn.bias.data = weights["conv2.1.bias"]
    #             cnn.conv2_bn.running_mean.data = weights["conv2.1.running_mean"]
    #             cnn.conv2_bn.running_var.data = weights["conv2.1.running_var"]
    #
    #             cnn.conv3.weight.data = weights["conv3.0.weight"]
    #             cnn.conv3_bn.weight.data = weights["conv3.1.weight"]
    #             cnn.conv3_bn.bias.data = weights["conv3.1.bias"]
    #             cnn.conv3_bn.running_mean.data = weights["conv3.1.running_mean"]
    #             cnn.conv3_bn.running_var.data = weights["conv3.1.running_var"]
    #
    #             cnn.conv3_1.weight.data = weights["conv3_1.0.weight"]
    #             cnn.conv3_1_bn.weight.data = weights["conv3_1.1.weight"]
    #             cnn.conv3_1_bn.bias.data = weights["conv3_1.1.bias"]
    #             cnn.conv3_1_bn.running_mean.data = weights["conv3_1.1.running_mean"]
    #             cnn.conv3_1_bn.running_var.data = weights["conv3_1.1.running_var"]
    #
    #             cnn.conv4.weight.data = weights["conv4.0.weight"]
    #             cnn.conv4_bn.weight.data = weights["conv4.1.weight"]
    #             cnn.conv4_bn.bias.data = weights["conv4.1.bias"]
    #             cnn.conv4_bn.running_mean.data = weights["conv4.1.running_mean"]
    #             cnn.conv4_bn.running_var.data = weights["conv4.1.running_var"]
    #
    #             cnn.conv4_1.weight.data = weights["conv4_1.0.weight"]
    #             cnn.conv4_1_bn.weight.data = weights["conv4_1.1.weight"]
    #             cnn.conv4_1_bn.bias.data = weights["conv4_1.1.bias"]
    #             cnn.conv4_1_bn.running_mean.data = weights["conv4_1.1.running_mean"]
    #             cnn.conv4_1_bn.running_var.data = weights["conv4_1.1.running_var"]
    #
    #             cnn.conv5.weight.data = weights["conv5.0.weight"]
    #             cnn.conv5_bn.weight.data = weights["conv5.1.weight"]
    #             cnn.conv5_bn.bias.data = weights["conv5.1.bias"]
    #             cnn.conv5_bn.running_mean.data = weights["conv5.1.running_mean"]
    #             cnn.conv5_bn.running_var.data = weights["conv5.1.running_var"]
    #
    #             cnn.conv5_1.weight.data = weights["conv5_1.0.weight"]
    #             cnn.conv5_1_bn.weight.data = weights["conv5_1.1.weight"]
    #             cnn.conv5_1_bn.bias.data = weights["conv5_1.1.bias"]
    #             cnn.conv5_1_bn.running_mean.data = weights["conv5_1.1.running_mean"]
    #             cnn.conv5_1_bn.running_var.data = weights["conv5_1.1.running_var"]
    #
    #             cnn.conv6.weight.data = weights["conv6.0.weight"]
    #             cnn.conv6_bn.weight.data = weights["conv6.1.weight"]
    #             cnn.conv6_bn.bias.data = weights["conv6.1.bias"]
    #             cnn.conv6_bn.running_mean.data = weights["conv6.1.running_mean"]
    #             cnn.conv6_bn.running_var.data = weights["conv6.1.running_var"]
    #
    #     return cnn
