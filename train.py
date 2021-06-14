import numpy as np
import os
import torch
import torch.nn.functional as F

from data import SpeechDataset, SpeechDataLoader, featurelen, cer, wer
from uyghur import uyghur_latin
from tqdm import tqdm
from UModel import UModel

from torch.optim.lr_scheduler import CosineAnnealingLR

class CustOpt:
    def __init__(self, params, datalen, lr, min_lr = None):
        if min_lr is None:
            min_lr = lr

        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.000001)  #, weight_decay=0.000001
        self._step = 0
        self.scheduler = CosineAnnealingLR(self.optimizer,T_max=datalen, eta_min = min_lr)

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
            outputs, output_lengths = model(inputs, input_lengths)
            loss = calctc_loss(outputs, targets, output_lengths, target_lengths)
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
        for pred, src in zip(preds, targets):
            e_char_cnt, char_cnt = cer(pred,src)
            e_word_cnt, word_cnt = wer(pred, src)
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

        outputs, output_lengths = model(inputs, input_lengths)
        loss = calctc_loss(outputs, targets, output_lengths, target_lengths)
        loss.backward()

        lr = optimizer.step()
        total_loss += loss.item()
        iter_cnt += 1

        msg = f'[LR: {lr: .7f} Loss: {loss.item(): .5f}, Avg loss: {(total_loss/iter_cnt): .5f}]'
        pbar.set_description(msg)
        if iter_cnt > mini_epoch_length:
            break
        
    pbar.close()
    with open(log_name,'a', encoding='utf-8') as fp:
        msg = f'Epoch[{(epoch+1):d}]:\t{msg}\n'
        fp.write(msg)


if __name__ == "__main__":
    device = "cuda"
    
    os.makedirs('./results',exist_ok=True)

    train_file = 'thuyg20_train.csv'
    test_file  = 'thuyg20_test.csv'
    
    train_set    = SpeechDataset(train_file, augumentation=True)
    train_loader = SpeechDataLoader(train_set,num_workers=4, pin_memory = True, shuffle=True, batch_size=20)

    validation_set = SpeechDataset(test_file, augumentation=False)
    validation_loader = SpeechDataLoader(validation_set,num_workers=4, pin_memory = True, shuffle=True, batch_size=20)

    print("="*50)
    msg =  f"        Training Set: {train_file}, {len(train_set)} samples" + "\n" 
    msg += f"      Validation Set: {test_file}, {len(validation_set)} samples" + "\n"
    msg += f"         Vocab Size : {uyghur_latin.vocab_size}" 

    print(msg)
    model = UModel(num_features_input = featurelen) 

    print("="*50)

    log_name = model.checkpoint + '.log'
    with open(log_name,'a', encoding='utf-8') as fp:
        fp.write(msg+'\n')

    model = model.to(device)

    #Start train and validation
    testfile=["test1.wav","test2.wav", "test3.wav","test4.wav","test5.wav","test6.wav"]
    start_epoch = model.trained_epochs
    mini_epoch_length = len(train_loader)

    optimizer = CustOpt(model.parameters(), mini_epoch_length, lr = 0.00002, min_lr=0.00002)
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
        if (epoch+1) % 2 == 0:
            print("Validating:")
            model.save((epoch+1))
            curcer = validate(model,validation_loader)
            if curcer < model.best_cer:
                model.best_cer = curcer
                model.save((epoch+1),best=True)

        model.save((epoch+1))
