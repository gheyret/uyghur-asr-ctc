import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from data import SpeechDataset, SpeechDataLoader, featurelen, cer_wer, cer, wer
from uyghur import uyghur_latin
from tqdm import tqdm


from GCGCResM import GCGCResM
from GCGCRes import GCGCRes
from GCGCRes1 import GCGCRes1
from GCGCRes2 import GCGCRes2
from QuartzNet import QuartzNet15x5, QuartzNet10x5, QuartzNet5x5
from UDS2W2L import UDS2W2L
from UDS2W2L3 import UDS2W2L3
from UDS2W2L5 import UDS2W2L5
from UDS2W2L50 import UDS2W2L50
from UDS2W2L8 import UDS2W2L8
from UDS2W2L80 import UDS2W2L80
#from FuncNet1 import FuncNet1
from UArilash0 import UArilash0
from UArilash1 import UArilash1

from UFormerCTC1 import UFormerCTC1
from UFormerCTC2 import UFormerCTC2
from UFormerCTC3 import UFormerCTC3
from UFormerCTC5 import UFormerCTC5
from UFormerCTC3N import UFormerCTC3N 
from uformer1dgru import UFormer1DGRU
from UFormerCTC1N import UFormerCTC1N

from ConfModelN import ConfModelN
from ConfModelM import ConfModelM
from ConfModelM2D import ConfModelM2D
from tiny_wav2letter import TinyWav2Letter
from UDS2W2L050 import UDS2W2L050

from UDeepSpeech import UDeepSpeech
from Conv1D3InDS2 import Conv1D3InDS2
from UDS2W2LGLU0 import UDS2W2LGLU0
from UDS2W2LGLU import UDS2W2LGLU
from UDS2W2LGLU8 import UDS2W2LGLU8

from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, StepLR
import random

from torch.cuda.amp import GradScaler

# Fix seed
# seed = 17
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)

class CustOpt:
    def __init__(self, params, datalen, lr, min_lr = None):
        if min_lr is None:
            min_lr = lr

        self.optimizer = torch.optim.Adam(params, lr=lr)  #, weight_decay=0.00001
        #self.optimizer = torch.optim.Adamax(params, lr=lr, weight_decay=0.00001)
        #self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay = 0.00001)
        #self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.00001)
        self._step = 0
        self.scheduler = CosineAnnealingLR(self.optimizer,T_max=datalen, eta_min = min_lr)
        #self.scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        #self.scheduler = CyclicLR(self.optimizer, T_max=datalen, eta_min = min_lr)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()
        rate = self.scheduler.get_last_lr()[0]
        return rate

    def zero_grad(self):
        self.optimizer.zero_grad()

#outputs format = B x F x T
def calctc_loss(outputs, targets, output_lengths, target_lengths):
    loss = F.ctc_loss(outputs.permute(2,0,1).contiguous(), targets, output_lengths, target_lengths, blank = uyghur_latin.pad_idx, reduction='mean',zero_infinity=True)
    return loss

def cal_loss(pred, gold):
    """
    Calculate metrics
    args:
        pred: B x T x C
        gold: B x T
        input_lengths: B (for CTC)
        target_lengths: B (for CTC)
    """
    gold = gold.contiguous().view(-1) # (B*T)
    pred = pred.contiguous().view(-1, pred.size(2)) # (B*T) x C
    loss = F.cross_entropy(pred, gold, ignore_index=uyghur_latin.pad_idx, reduction="mean")
    return loss


def validate(model, valid_loader):
    chars = 0
    words = 0
    e_chars = 0
    e_words = 0
    avg_loss = 0
    iter_cnt = 0
    msg = ""
    
    cer_val = 0.0

    model.eval()
    with torch.no_grad():
        tlen = len(valid_loader)
        vbar = tqdm(iter(valid_loader), leave=True, total=tlen)
        for inputs, targets, input_lengths, target_lengths, _ in vbar:

            inputs  = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            if model_type == 'CTC':
                outputs, output_lengths = model(inputs, input_lengths)
                loss = calctc_loss(outputs, targets, output_lengths, target_lengths)
            elif model_type =='S2S':
                output_lengths = 0
                outputs, tgt = model(inputs, input_lengths, targets)
                loss = cal_loss(outputs, tgt)
            elif model_type == 'JOINT':
                output_lengths = 0
                outputs, tgt = model(inputs, input_lengths, targets)
                loss1 = cal_loss(outputs, tgt)
                loss_ctc= calctc_loss(model.ctcOut, targets, model.ctcLen, target_lengths)
                #loss = loss1*0.6 + loss_ctc*0.4
                loss = loss1*0.78 + loss_ctc*0.22
                #loss = loss1*0.22 + loss_ctc*0.78

            preds   = model.greedydecode(outputs, output_lengths)
            targets = [uyghur_latin.decode(target) for target in targets]
            
            for pred, src in zip(preds, targets):
                e_char_cnt, char_cnt = cer(pred,src)
                e_word_cnt, word_cnt = wer(pred, src)
                e_chars += e_char_cnt
                e_words += e_word_cnt

                chars += char_cnt
                words += word_cnt

            iter_cnt += 1
            avg_loss +=loss.item()

            msg = f"  VALIDATION: [CER:{e_chars/chars:.2%} ({e_chars}/{chars} letters) WER:{e_words/words:.2%} ({e_words}/{words} words), Avg loss:{avg_loss/iter_cnt:4f}]"
            vbar.set_description(msg)

        vbar.close()

        cer_val = e_chars/chars

        with open(log_name,'a', encoding='utf-8') as fp:
            fp.write(msg+"\n")

        #Print Last 3 validation results
        result =""
        result_cnt = 0
        chars = 0
        words = 0
        e_chars = 0
        e_words = 0
        for pred, src in zip(preds, targets):
            e_char_cnt, char_cnt = cer(pred,src)
            e_word_cnt, word_cnt = wer(pred, src)
            e_chars += e_char_cnt
            e_words += e_word_cnt
            chars += char_cnt
            words += word_cnt
            result += f"   O:{src}\n"
            result += f"   P:{pred}\n"
            result += f"     CER: {e_char_cnt/char_cnt:.2%} ({e_char_cnt}/{char_cnt} letters), WER: {e_word_cnt/word_cnt:.2%} ({e_word_cnt}/{word_cnt} words)\n"
            result_cnt += 1
            if result_cnt >= 3:
                break
        
        print(result)
        return cer_val


def train(model, train_loader):
    total_loss = 0
    iter_cnt = 0
    msg =''
    model.train()
    pbar = tqdm(iter(train_loader), leave=True, total=mini_epoch_length)
    for data in pbar:
        optimizer.zero_grad()
        inputs, targets, input_lengths, target_lengths, _ = data
        inputs  = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        if model_type == 'CTC':
            outputs, output_lengths = model(inputs, input_lengths)
            loss = calctc_loss(outputs, targets, output_lengths, target_lengths)
        elif model_type =='S2S':
            output_lengths = 0
            outputs, tgt = model(inputs, input_lengths, targets)
            loss = cal_loss(outputs, tgt)
        elif model_type == 'JOINT':
            output_lengths = 0
            outputs, tgt = model(inputs, input_lengths, targets)
            loss1 = cal_loss(outputs, tgt)
            loss_ctc = calctc_loss(model.ctcOut, targets, model.ctcLen, target_lengths)
            #loss = loss1*0.6 + loss_ctc*0.4
            loss = loss1*0.78 + loss_ctc*0.22
            #loss = loss1*0.22 + loss_ctc*0.78

        loss.backward()
        lr = optimizer.step()
        total_loss += loss.item()
        iter_cnt += 1

        msg = f'[LR: {lr: .6f} Loss: {loss.item(): .5f}, Avg loss: {(total_loss/iter_cnt): .5f}]'
        pbar.set_description(msg)
        #torch.cuda.empty_cache()
        if iter_cnt > mini_epoch_length:
            break
        
    pbar.close()
    with open(log_name,'a', encoding='utf-8') as fp:
        msg = f'Epoch[{(epoch+1):d}]:\t{msg}\n'
        fp.write(msg)

def GetModel():

    if model_type == 'CTC':
        #model = GCGCResM(num_features_input = featurelen)  
        #model = UDS2W2L(num_features_input = featurelen)        
        #model = GCGCRes2(num_features_input = featurelen) 
        #model = GCGCRes(num_features_input = featurelen)      # Bashqa yerde mengiwatidu
        #model = GCGCRes1(num_features_input = featurelen)      # Bashqa yerde mengiwatidu

        #model = UDS2W2L50(num_features_input = featurelen) 
        #model = UDS2W2L80(num_features_input = featurelen)
        #model  = ConfModel(num_features_input = featurelen)

        #model = QuartzNet15x5(num_features_input = featurelen)
        #model = QuartzNet10x5(num_features_input = featurelen)
        #model = QuartzNet5x5(num_features_input = featurelen)

        #model = UArilash1(num_features_input = featurelen)
        #model = UDeepSpeech(num_features_input = featurelen)
        #model = UDS2W2L3(num_features_input = featurelen)        


        #model = TinyWav2Letter(num_features_input = featurelen)  
        #model  = ConfModelM(num_features_input = featurelen)

        #model = UDS2W2L050(num_features_input = featurelen)
        #model  = Conv1D3InDS2(num_features_input = featurelen)
        #model  = UDS2W2LGLU(num_features_input = featurelen)
        model  = UDS2W2LGLU8(num_features_input = featurelen)

    elif model_type == 'S2S':
        #model = UFormer(num_features_input = featurelen)
        #model = UFormer1DGRU(num_features_input = featurelen)
        
        #model = UFormerCTC(num_features_input = featurelen)
        #model = UFormerCTC3(num_features_input = featurelen)
        model = UFormerCTC3N(num_features_input = featurelen)
        #model = UFormerCTC1N(num_features_input = featurelen)

    elif model_type =='JOINT':
        #model = UFormer(num_features_input = featurelen)
        #model = UFormer1DGRU(num_features_input = featurelen)

        #model = UFormerCTC(num_features_input = featurelen)
        #model = UFormerCTC3(num_features_input = featurelen)
        #model = UFormerCTC3N(num_features_input = featurelen)
        model = UFormerCTC1N(num_features_input = featurelen)
    

    return model


#Sinaydighan modellar
#UFormerCTC3N
#UDS2W2L5
#GCGCRes1

if __name__ == "__main__":
    device = "cuda"
    os.makedirs('./results',exist_ok=True)

    model_type = 'CTC' # S2S, 'JOINT', 'CTC'
    
    #train_file = 'uyghur_train.csv'
    train_file = 'uyghur_thuyg20_train_small.csv'
    test_file  = 'uyghur_thuyg20_test_small.csv'
    
    train_set = SpeechDataset(train_file, augumentation=False)
    train_loader = SpeechDataLoader(train_set,num_workers=5, pin_memory = True, shuffle=True, batch_size=24)

    validation_set = SpeechDataset(test_file, augumentation=False)
    validation_loader = SpeechDataLoader(validation_set,num_workers=5, pin_memory = True, shuffle=True, batch_size=24)

    print("="*50)
    msg =  f"        Training Set: {train_file}, {len(train_set)} samples" + "\n" 
    msg += f"      Validation Set: {test_file}, {len(validation_set)} samples" + "\n"
    msg += f"         Vocab Size : {uyghur_latin.vocab_size}" 

    print(msg)
    model = GetModel()
    print("="*50)

    log_name = model.checkpoint + '.log'
    with open(log_name,'a', encoding='utf-8') as fp:
        fp.write(msg+'\n')

    train_set.Raw = model.Raw       #If it using RAW wave form data
    validation_set.Raw = model.Raw  #If it using RAW wave form data

    model = model.to(device)

    #Star train and validation
    testfile=["test1.wav","test2.wav", "test3.wav","test4.wav","test5.wav","test6.wav"]
    start_epoch = model.trained_epochs
    mini_epoch_length = len(train_loader)
    if mini_epoch_length > 1000:
        mini_epoch_length = mini_epoch_length//2
        #pass

    optimizer = CustOpt(model.parameters(), mini_epoch_length//2, lr = 0.0001, min_lr=0.00001)
    for epoch in range(start_epoch,1000):
        torch.cuda.empty_cache()
        model.eval()
        msg = ""
        for afile in testfile:
            text = model.predict(afile,device)
            text = f"{afile}-->{text}\n"
            print(text,end="")
            msg += text

        with open(log_name,'a', encoding='utf-8') as fp:
            fp.write(msg+'\n')

        print("="*50)
        print(f"Training Epoch[{(epoch+1):d}]:")
        train(model, train_loader)
        if (epoch+1) % 1 == 0:
            print("Validating:")
            model.save((epoch+1))
            curcer = validate(model,validation_loader)
            if curcer < model.best_cer:
                model.best_cer = curcer
                model.save((epoch+1),best=True)

        model.save((epoch+1))
