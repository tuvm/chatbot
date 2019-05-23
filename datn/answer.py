# coding: utf-8
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
from random import sample
from seq2seq_model import *
from preprocess_data import Data
from underthesea import word_tokenize
import pickle

class Answer(object):
    def __init__(self, emb_dim=512, ckpt_path='ckpt512/', num_layers=3):
        self.data = Data()
        metadata, idx_q, idx_a = self.data.load_data()

        # parameters
        xseq_len = metadata['limit']['maxq']
        yseq_len = metadata['limit']['maxa']
        batch_size = 32
        xvocab_size = len(metadata['idx2w'])
        yvocab_size = xvocab_size

        self.model = Seq2Seq(xseq_len=xseq_len,
            yseq_len=yseq_len,
            xvocab_size=xvocab_size,
            yvocab_size=yvocab_size,
            ckpt_path=ckpt_path,
            emb_dim=emb_dim,
            num_layers=num_layers
        ) # create model

        self.word2index = metadata['w2idx']
        self.index2word = metadata['idx2w']
        self.sess = self.model.restore_last_session() # restore session to predict

    def answer(self, sentence):
        """
            Using answer to predict reply from sentence - text from user
            This function is used in ui/ui.py
        """
        test_sentence = sentence.decode('utf-8')
        test_sentence = word_tokenize(test_sentence, format="text").split()
        sentence2index = self.data.zero_pad([test_sentence], [''], self.word2index)
        sentence_input = sentence2index[0].T
        sentence_output = sentence2index[1].T

        predict_result = self.model.predict(self.sess, sentence_input, sentence_output, using_attention=2)
        predict_result = predict_result[0]
        predict_sentence = []
        predict_words = []
        for predict_word in predict_result: # postprocess result to remove same word together
            if predict_word == 0:
                continue
            if predict_sentence and predict_word == predict_sentence[-1]:
                continue
            predict_sentence.append(predict_word)
            predict_words.append(self.index2word[predict_word])
        predict_words = ' '.join(predict_words)
        return predict_words

if __name__ == '__main__':
    answer = Answer()
    answer.answer('TYPE còn những COLO nào ạ')
