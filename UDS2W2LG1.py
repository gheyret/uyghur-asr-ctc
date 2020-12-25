import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from data import melfuture
from uyghur import uyghur_latin
from BaseModel import BaseModel


class UDS2W2LG1(BaseModel):
    def __init__(self,num_features_input,load_best=False):
        super(UDS2W2LG1, self).__init__('UDS2W2LG1')
        dropout = 0.1
        #UDS2W2LG ning bu yerila ozgerdi
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5),bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )
        self.lstm1 = nn.GRU(1024, 256, num_layers=1 , batch_first=True, bidirectional=True)
        self.cnn1  = nn.Sequential(
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2)
        )
        self.lstm2 = nn.GRU(256, 384, num_layers=2 , batch_first=True, bidirectional=True)
        self.cnn2 = nn.Sequential(
            ResB(384,13,6,0.2),
            ResB(384,13,6,0.2),
            ResB(384,13,6,0.2),
            nn.Conv1d(384, 512, 17, 1,8,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(512,17,8,0.3),
            ResB(512,17,8,0.3),
            nn.Conv1d(512, 1024, 1, 1,bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResB(1024,1,0,0.0),
        )
        self.outlayer = nn.Conv1d(1024, uyghur_latin.vocab_size, 1, 1)
        self.softMax = nn.LogSoftmax(dim=1)

        self.checkpoint = 'results/' + self.ModelName
        self._load()
        print(f'The model has {self.parameters_count(self):,} trainable parameters')

    def smooth_labels(self, x):
        return (1.0 - self.smoothing) * x + self.smoothing / x.size(-1)

    def forward(self, x, lengths):
        out_lens = lengths//4

        x.unsqueeze_(1)
        out = self.conv(x)

        b, c, h, w = out.size()
        out = out.view(b, c*h, w).contiguous() #.permute(0,2,1)

        out = out.permute(0,2,1)
        out = nn.utils.rnn.pack_padded_sequence(out, out_lens, batch_first=True)
        out, _ = self.lstm1(out)        
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = (out[:, :, :self.lstm1.hidden_size] + out[:, :, self.lstm1.hidden_size:]).contiguous()
        out = self.cnn1(out.permute(0,2,1))

        out = out.permute(0,2,1)
        out = nn.utils.rnn.pack_padded_sequence(out, out_lens, batch_first=True)
        out,_ = self.lstm2(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = (out[:, :, :self.lstm2.hidden_size] + out[:, :, self.lstm2.hidden_size:]).contiguous()
        out = self.cnn2(out.permute(0,2,1))
        out = self.outlayer(out)
        out = self.softMax(out)
        return out, out_lens


class ResB(nn.Module):
    def __init__(self, num_filters, kernel, pad, d = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size = kernel, stride = 1 , padding=pad, bias=False),
            nn.BatchNorm1d(num_filters)
            )

        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm1d(num_filters)
        self.drop =nn.Dropout(d)

    def forward(self, x):
        identity = x
        out  = self.conv(x)
        out += identity
        out  = self.bn(out)
        out  = self.relu(out)
        out  = self.drop(out)
        return out



if __name__ == "__main__":
    from data import featurelen, melfuture
    device ="cpu"

    net = UDS2W2LG1(featurelen).to(device)
    text = net.predict("test1.wav",device)
    print(text)
    text = net.predict("test2.wav",device)
    print(text)


    #net.best_cer = 1.0
    #net.save(0)


    melf = melfuture("test3.wav")
    melf.unsqueeze_(0)

    conv0 = nn.Conv1d(featurelen,256,11,2, 5, 1)

    conv1 = nn.Conv1d(256,256,11,1, 5, 1)
    conv3 = nn.Conv1d(256,256,11,1, 5*2, 2)
    conv5 = nn.Conv1d(256,256,11,1, 5*3, 3)

    out0 = conv0(melf)
 
    out1 = conv1(out0)
    out3 = conv3(out0)
    out5 = conv5(out0)

    print(out1.size())
    print(out3.size())
    print(out5.size())

    out = out1 * out3 * out5
    print(out.size())


    #net = GCGCRes(featurelen).to(device)
    #net.save(1)

    #text = net.predict("test1.wav",device)
    #print(text)
    #text = net.predict("test2.wav",device)
    #print(text)
