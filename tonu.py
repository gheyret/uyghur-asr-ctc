import sys
import os
from data import featurelen
from UModel import UModel

if __name__ == '__main__':
    model = UModel(featurelen)
    
    if len(sys.argv)<2:
        print("Using \n\tpython tonu.py audiofile | audiolist.txt | folder")
    else:
        device = 'cpu'
        model.to(device)
        audiofile = sys.argv[1]
        print(f"\n======================\nRecognizing file {audiofile}")
        txt = model.predict(audiofile,device)
        print("%s -> %s" %(os.path.basename(audiofile),txt))
