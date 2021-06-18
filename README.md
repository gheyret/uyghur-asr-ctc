# Speech Recognition for Uyghur using deep learning
Training:

this model using CTC loss for training.

Download [pretrained model](https://github.com/gheyret/uyghur-asr-ctc/releases/download/data/results.7z) and [dataset](https://github.com/gheyret/uyghur-asr-ctc/releases/download/data/thuyg20_data.7z).

unzip results.7z and thuyg20_data.7z to the same folder where python source files located. then run:
```
python train.py
```

Recognition:

for recognition download only pretrained model(results.7z). then run:

```
python tonu.py test1.wav 
```
result will be:
```
        Model loaded: results/UModel_last.pth
            Best CER: 7.21%
             Trained: 473 epochs
The model has 26,389,282 trainable parameters

======================
Recognizing file .\test2.wav
test2.wav -> bu öy eslide xotunining xush tebessumi oghlining omaq külküsi bilen güzel idi
```

This project using 

[**A free Uyghur speech database Released by CSLT@Tsinghua University & Xinjiang University**](http://www.openslr.org/22/)

