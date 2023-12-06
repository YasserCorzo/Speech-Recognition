from model import KeywordClassification
from hmm import HMM
from pathlib import Path
import numpy as np
import random

if __name__ == '__main__':

    state_size = 10
    component_size = 3
    feature_size = 40
    vocab_size = 2

    model = KeywordClassification(state_size, component_size, feature_size, vocab_size)

    X_train = []
    y_train = []
    
    for line in open('./train.txt', 'r'):
        filename, label = line.strip().split()
        feat = np.load('./train/'+filename, allow_pickle=True)
        label = int(label)

        X_train.append(feat)
        y_train.append(label)

    X_test = []
    filename_lst = []
    
    for file in sorted(Path('./test/').glob('*')):
        feat = np.load(str(file), allow_pickle=True)
        filename_lst.append(file.name)
        X_test.append(feat)

    model.fit(X_train, y_train)

    y_test = model.predict(X_test)

    output = open('./test.txt', 'w')

    for filename, y in zip(filename_lst, y_test):
        output.write(filename+' '+str(y)+'\n')

    output.close()
