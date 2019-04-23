import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_CIFAR10(file):
    dataTrain = []
    labelTrain = []
    for i in range(1,5):
        dic = unpickle(file+'/data_batch_'+str(i))
        for item in dic['data']:
            dataTrain.append(item)
        for item in dic['labels']:
            labelTrain.append(item)

    dataTest = []
    labelTest = []
    dic = unpickle(file+'/test_batch')
    for item in dic['data']:
        dataTest.append(item)
    for item in dic['labels']:
        labelTest.append(item)
    return (dataTrain,labelTrain,dataTest,labelTest)

datatr, labeltr, datate, labelte = load_CIFAR10('/media/alex/70121F27121EF1B8/cifar-10-batches-py')
Xtr = np.asarray(datatr)
Xte = np.asarray(datate)
Ytr = np.asarray(labeltr)
Yte = np.asarray(labelte)
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
