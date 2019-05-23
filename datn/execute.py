import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
from random import sample
from seq2seq_model import *
from preprocess_data import Data
import pickle

def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    """
        split data into train (70%), test (15%) and valid(15%)
    """
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)

def batch_gen(x, y, batch_size):
    """
        generate batches from dataset
        yield (x_gen, y_gen)
    """
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

def rand_batch_gen(x, y, batch_size):
    """
        generate batches, by random sampling a bunch of items
        yield (x_gen, y_gen)

    """
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    """
        decode from index to word
    """
    return separator.join([lookup[element] for element in sequence if element])

def execute():
    data = Data()
    metadata, idx_q, idx_a = data.load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)

    # parameters
    xseq_len = trainX.shape[-1]
    yseq_len = trainY.shape[-1]
    batch_size = 32
    xvocab_size = len(metadata['idx2w'])
    yvocab_size = xvocab_size
    emb_dim = 512

    model = Seq2Seq(xseq_len=xseq_len,
           yseq_len=yseq_len,
           xvocab_size=xvocab_size,
           yvocab_size=yvocab_size,
           ckpt_path='ckpt512/',
           emb_dim=emb_dim,
           num_layers=3
           )

    val_batch_gen = rand_batch_gen(validX, validY, 32)
    train_batch_gen = rand_batch_gen(trainX, trainY, batch_size)
    # test_batch_gen = rand_batch_gen(testX, testY, 1)

    sess = model.restore_last_session()
    sess = model.train(train_batch_gen, val_batch_gen, sess, using_attention=2)
    # print '\n'
    # for i in range(10):
    #     print i
    #     test_data = test_batch_gen.next()
    #     test_question = decode(test_data[0].T[0], metadata['idx2w'], ' ')
    #     print test_question
    #     test_output = np.zeros([1, 50], dtype=np.int32).T
    #     predict_result = model.predict(sess, test_data[0], test_output, using_attention=2)
    #     predict_answer = decode(predict_result[0], metadata['idx2w'], ' ')
    #     print predict_answer

if __name__ == '__main__':
    execute()
