import torch
from torch import nn

class STCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv_block = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=0.5)
        self.max_pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        return self.max_pool(self.dropout(self.relu(self.batch_norm(self.conv_block(x)))))


class BiGRU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=output_size, batch_first=True, bidirectional=True, num_layers=2)

    def forward(self, x):
        batch_size, frames, channel, h, w = x.shape
        x = x.reshape(batch_size, frames, channel*h*w)
        return self.gru(x)


class LipNet(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.stcnn_1 = STCNN(in_channels=in_channels, out_channels=32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.stcnn_2 = STCNN(in_channels=32, out_channels=64, kernel_size=(3,5,5), stride=(1,1,1), padding=(1,2,2))
        self.stcnn_3 = STCNN(in_channels=64, out_channels=96, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        self.gru_1 = BiGRU(input_size=96*6*6, output_size=256)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=29*512, out_features=output_size)

    def forward(self, x):
        x = self.stcnn_1(x)
        x = self.stcnn_2(x)
        x = self.stcnn_3(x)

        x = x.permute(0,2,1,3,4)
        x,_ = self.gru_1(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

