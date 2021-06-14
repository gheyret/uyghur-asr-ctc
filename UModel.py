import os
import torch
import torch.nn as nn
from uyghur import uyghur_latin
from data import melfuture

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

class UModel(nn.Module):
    def __init__(self, num_features_input, load_best=False):
        super(UModel, self).__init__()

        self.in1 = nn.Conv1d(128,256,11,2, 5*1, dilation = 1, bias=False)
        self.in2 = nn.Conv1d(128,256,15,2, 7*2, dilation = 2, bias=False)
        self.in3 = nn.Conv1d(128,256,19,2, 9*3, dilation = 3, bias=False)
        self.concat  = nn.Conv1d(256*3,256,1,1,bias=True)
        self.relu = nn.ReLU()

        self.cnn1  = nn.Sequential(
            nn.Conv1d(256, 256, 11, 1, 5, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2)
        )
        self.rnn = nn.GRU(256, 384, num_layers=1 , batch_first=True, bidirectional=True)
        self.cnn2 = nn.Sequential(
            ResB(384,13,6,0.2),
            ResB(384,13,6,0.2),
            ResB(384,13,6,0.2),
            nn.Conv1d(384, 512, 17, 1,8, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(512,17,8,0.3),
            ResB(512,17,8,0.3),
            nn.Conv1d(512, 1024, 1, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResB(1024,1,0,0.0),
        )
        self.outlayer = nn.Conv1d(1024, uyghur_latin.vocab_size, 1, 1)
        self.softMax = nn.LogSoftmax(dim=1)

        self.checkpoint = 'results/UModel'
        self._load(load_best)
        print(f'The model has {self.parameters_count(self):,} trainable parameters')

    # X : N x F x T
    def forward(self, x, input_lengths):

        inp = torch.cat([self.in1(x), self.in2(x), self.in3(x)],dim = 1)
        inp = self.concat(inp)
        inp = self.relu(inp)
        out = self.cnn1(inp)

        out_lens = input_lengths//2
        out = out.permute(0,2,1)

        out,_ = self.rnn(out)
        out = (out[:, :, :self.rnn.hidden_size] + out[:, :, self.rnn.hidden_size:]).contiguous()

        out = self.cnn2(out.permute(0,2,1))
        out = self.outlayer(out)
        out = self.softMax(out) 
        return out, out_lens


    def parameters_count(self, model):
        sum_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum_par

    def _load(self, load_best=False):
        path = None
        if load_best == True and os.path.exists(self.checkpoint + '_best.pth'):
            path = path = self.checkpoint + '_best.pth'
        elif os.path.exists(self.checkpoint + '_last.pth'):
            path = self.checkpoint + '_last.pth'
        
        if path is not None:
            pack = torch.load(path, map_location='cpu')
            self.load_state_dict(pack['st_dict'])
            self.trained_epochs = pack['epoch']
            self.best_cer = pack.get('BCER', 1.0)
            print(f'        Model loaded: {path}')
            print(f'            Best CER: {self.best_cer:.2%}')
            print(f'             Trained: {self.trained_epochs} epochs')

    def save(self, epoch, best = False):
        pack = {
            'st_dict':self.state_dict(),
            'epoch':epoch,
            'BCER':self.best_cer
            }

        if best == True:
            path = path = self.checkpoint + '_best.pth'
        else:
            path = path = self.checkpoint + '_last.pth'
        torch.save(pack, path)


    def predict(self, path, device):
        self.eval()
        spect = melfuture(path).to(device)    
        spect.unsqueeze_(0)
        xn = [spect.size(2)]
        xn = torch.IntTensor(xn)
        out, xn = self.forward(spect, xn)
        text = self.greedydecode(out, xn)
        self.train()
        return text[0]

    #CTC greedy decode
    def greedydecode(self, yps, yps_lens):
        _, max_yps = torch.max(yps, 1)
        preds = []
        for x in range(len(max_yps)):
            pred = []
            last = None
            for i in range(yps_lens[x]):
                char = int(max_yps[x][i].item())
                if char != uyghur_latin.pad_idx:
                    if char != last:
                        pred.append(char)
                last = char
            preds.append(pred)

        predstrs = [uyghur_latin.decode(pred) for pred in preds]
        return predstrs


if __name__ == "__main__":
    from data import featurelen, melfuture
    device ="cpu"

    net = UModel(featurelen).to(device)
    #net.save(0)

    text = net.predict("test1.wav",device)
    print(text)
    text = net.predict("test2.wav",device)
    print(text)

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
