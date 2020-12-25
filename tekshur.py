import torch
from data import SpeechDataset, SpeechDataLoader, featurelen, uyghur_latin, cer
from GCGCResM import GCGCResM
from uformer import UFormer
from UDS2W2L50 import UDS2W2L50
from UFormerCTC2 import UFormerCTC2

import sys
import os
import glob
from tqdm import tqdm

def tekshurctc(model, hojjet, device):
    training_set = SpeechDataset(hojjet, augumentation=False)
    loader = SpeechDataLoader(training_set,num_workers=4, shuffle=False, batch_size=32)

    line = []
    with torch.no_grad():
        pbar = tqdm(iter(loader), leave=True, total=len(loader))
        for inputs, targets, input_lengths, _ , paths in pbar:

            inputs  = inputs.to(device,non_blocking=True)
            outputs, output_lengths = model(inputs, input_lengths)
            preds   = model.greedydecode(outputs, output_lengths)
            targets = [uyghur_latin.decode(target) for target in targets]
            
            for pred, src, wavename in zip(preds, targets, paths):
                xatasani , _ = cer(pred, src)
                if xatasani >= 1:
                    xata = f"{wavename}\t{src}\t{xatasani}\n"
                    #xata = f"{src}\n{pred}\n\n"
                    line.append(xata)
    return line
    

def tekshurs2s(model, hojjet, device):
    training_set = SpeechDataset(hojjet, augumentation=False)
    loader = SpeechDataLoader(training_set,num_workers=4, shuffle=False, batch_size=20)

    line = []
    with torch.no_grad():
        pbar = tqdm(iter(loader), leave=True, total=len(loader))
        for inputs, targets, input_lengths, _ , paths in pbar:

            inputs  = inputs.to(device,non_blocking=True)
            targets  = targets.to(device,non_blocking=True)
            input_lengths  = input_lengths.to(device,non_blocking=True)

            outputs, _ = model(inputs, input_lengths, targets)
            preds   = model.greedydecode(outputs, 0)
            targets = [uyghur_latin.decode(target) for target in targets]
            
            for pred, src, wavename in zip(preds, targets, paths):
                xatasani , _ = cer(pred, src)
                if xatasani >= 5:
                    xata = f"{wavename}\t{src}\t{xatasani}\n"
                    #xata = f"{src}\n{pred}\n\n"
                    line.append(xata)
    return line

if __name__ == '__main__':
    device = 'cuda'
    #model = GCGCResM(featurelen, load_best=False)
    #model = UFormer(featurelen, load_best=False)
    
    model = UDS2W2L50(featurelen, load_best=False)
    #model = UFormerCTC2(featurelen, load_best=False)
    model.to(device)
    model.eval()

    #'uyghur_train.csv' 'uyghur_thuyg20_train_small.csv', ''
    #netije = tekshurs2s(model, 'uyghur_train.csv', device)
    netije = tekshurctc(model, 'uyghur_thuyg20_test_small.csv', device)
    with open('tek_test.csv','w',encoding='utf_8_sig') as f:
        f.writelines(netije)
