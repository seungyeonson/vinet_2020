#model
# File to return the Deep VO model.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from utils.util import correlate, conv

# DeepVO model
class VINet(nn.Module):

    def __init__(self, imageWidth, imageHeight, activation='relu', batchnorm=False, \
                 dropout=0.0, flownet_weights_path=None, numLSTMCells=1, hidden_units_imu=None, hidden_units_LSTM=None, \
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
        self.batchNorm = batchnorm

        # Whether or not dropout is required
        if dropout <= 0.0:
            self.dropout = False
        else:
            # Specify the drop_ratio
            self.dropout = True
            self.drop_ratio = dropout

        self.numLSTMCells = numLSTMCells
        self.hidden_units_LSTM = hidden_units_LSTM
        self.hidden_units_imu = hidden_units_imu

        # Path to FlowNet weights
        if flownet_weights_path is not None:
            self.use_flownet = True
            self.flownet_weights_path = flownet_weights_path
        else:
            self.use_flownet = False

        """False
        Initialize variables required for the network
        """

        # If we're using batchnorm, do not use bias for the conv layers
        self.bias = not self.batchNorm


        self.conv1 = conv(self.batchNorm, 1, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)


        self.rnnIMU = nn.LSTM(
            input_size=6,
            hidden_size=6, #hidden_units_imu[0],
            num_layers=2,
            batch_first=True
        )
        self.rnnIMU.cuda()

        self.rnn = nn.LSTM(
                input_size=1037,
                hidden_size=hidden_units_LSTM[0],
                num_layers =2,
                batch_first=True
        )
        self.rnn.cuda()

        self.fc1 = nn.Linear(self.hidden_units_LSTM[self.numLSTMCells - 1], 128)
        self.fc1.cuda()
        # self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc2.cuda()

        self.fc_out = nn.Linear(32, 6)
        self.fc_out.cuda()



    def forward(self, x, imu, xyzq, reset_hidden=False):
        if not self.batchNorm:
            x1 = x[:,0:1,:,:]
            x2 = x[:,1:,:,:]

            x1 = self.conv1(x1)
            x1 = self.conv2(x1)
            x1 = self.conv3(x1)

            x2 = self.conv1(x2)
            x2 = self.conv2(x2)
            x2 = self.conv3(x2)

            redir = self.conv_redir(x1)
            cor = correlate(x1,x2)
            x = torch.cat([redir, cor], dim=1)

            x = self.conv3_1(x)
            x = self.conv4(x)
            x = self.conv4_1(x)
            x = self.conv5(x)
            x = self.conv5_1(x)
            # x = (F.leaky_relu(self.conv(x)))
            x = self.conv6(x)
            x = self.conv6_1(x)
            # imu = imu.view(11,1,6)

            # if reset_hidden is True:
            #
            #     self.h1 = torch.zeros(1, 6)
            #     self.c1 = torch.zeros(1, 6)
            #     self.h2 = torch.zeros(1, 6)
            #     self.c2 = torch.zeros(1, 6)
            # for i in range(11):
            #     self.h1, self.c1 = self.rnnIMU(imu[i], (self.h1, self.c1))
            #     self.h2, self.c2 = self.rnnIMU(self.h1, (self.h2, self.c2))

            imu_out, (imu_n,imu_c) = self.rnnIMU(imu)
            imu_out = imu_out[:,-1,:]
            imu_out = imu_out.unsqueeze(1)

            r_in = x.view(1, 1, -1)
            r_in = torch.cat((r_in, imu_out), 2)
            r_in = torch.cat((r_in, xyzq), 2)

            r_out, (h_n, h_c) = self.rnn(r_in)


            # Forward pass through the FC layers
            if self.activation == 'relu':
                if self.numLSTMCells == 1:
                    output_fc1 = F.relu(self.fc1(r_out))
                else:
                    output_fc1 = F.relu(self.fc1(r_out))
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
                    output_fc1 = F.selu(self.fc1(r_out))
                else:
                    output_fc1 = F.selu(self.fc1(r_out))
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

    # def detach_LSTM_hidden(self):
    #     self.h1 = self.h1.detach()
    #     self.c1 = self.c1.detach()
    #     self.h2 = self.h2.detach()
    #     self.c2 = self.c2.detach()

    # def reset_LSTM_hidden(self):
    #     self.h1 = torch.zeros(1, 11, self.hidden_units_imu[0])
    #     self.c1 = torch.zeros(1, 11, self.hidden_units_imu[0])
    #     self.h2 = torch.zeros(1, 1, self.hidden_units_LSTM[0])
    #     self.c2 = torch.zeros(1, 1, self.hidden_units_LSTM[0])
