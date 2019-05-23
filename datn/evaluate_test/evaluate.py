# coding: utf-8
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
from random import sample
from seq2seq_model import *
from preprocess_data import Data
from underthesea import word_tokenize
import pickle
from execute import decode
import answer
import csv

with open("./evaluate_test/question_test.txt", "r") as file:
    test_set = file.readlines()
test_set = [test_line.replace("\n", "") for test_line in test_set]

def evaluate(param, file_name):
    """
        Evaluate 4 parameter from question_test.txt and write result to file csv
    """
    print("======= Start Evaluate =======")
    print(param[1])
    answer_obj = answer.Answer(param[0], param[1], param[2])
    with open(file_name, "w") as csvfile:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, sentence in enumerate(test_set):
            data = answer_obj.answer(sentence)
            writer.writerow({ 'question': sentence, 'answer': data })
            print str(index) + ": " + sentence + "-----" + data
    print("======= End Of Evaluate =======")

if __name__ == '__main__':
    parameters = [
        [128, "ckpt128_attention", 3],
        [512, "ckpt512_attention", 3],
        [512, "ckpt512_attention_1_layer", 1]
    ]
    param_no_attention = [512, "ckpt512", 3]

    # Method in seq2seq_model.py must be embedding_attention_seq2seq
    for param in parameters:
        param = parameters[1]
        evaluate(param, "./evaluate_test/" + param[1] + ".csv")

    # Method in seq2seq_model.py must be embedding_rnn_seq2seq
    # evaluate(param_no_attention, "./evaluate_test/" + param[1] + ".csv")
