import re
from underthesea import word_tokenize
import sys
import os
import pickle
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

NER_LABELS = [
    "COLO",
    "MATE",
    "SIZE",
    "PRIC",
    "GEND",
    "ORIG",
    "TRAD",
    "TYPE",
    "LOC",
    "SAOF",
    "CURU",
    "SHME",
    "TIME",
    "REFE",
    "WEIG",
    "HEIG",
]
MULTI_SPACE = re.compile("\s+")
SPECIAL = re.compile('[?|$|\#|!|>|<|=|@|%|^|&|(|)|\"|,|\'|-|:|;|+|*|/|.]')

class Data(object):
    """
        Build class Data in order to preprocess raw data, build vocabulary,
            handle data before using it to train or test
    """

    def __init__(self):
        self.limit = {
            'maxq' : 50,
            'minq' : 0,
            'maxa' : 50,
            'mina' : 3
        }
        self.UNK = 'unk'

    def remove_multi_space(self, string):
        return re.sub(MULTI_SPACE, " ", string).strip()

    def remove_special_char(self, string):
        return re.sub(SPECIAL, "", string).strip()

    def preprocess(self, string):
        string = self.remove_special_char(string)
        string = self.remove_multi_space(string)

        return string

    def get_tokenizer(self, sentence):
        words = []
        sentence = self.remove_multi_space(sentence)
        for word in sentence.split(" "):
            if "<B-" in word:
                string = word[3:(len(word) - 1)]
                if string not in words:
                    word = string
                else:
                    continue
            elif "<I-" in word:
                continue
            else:
                if word.isalpha():
                    word = word.lower()
                elif word.isdigit():
                    word = "DIGIT"
                else:
                    word = "CODE"

            words.append(self.preprocess(word))

        words = " ".join(words)
        words = word_tokenize(words, format="text")

        return words

    def get_data_from_files(self, directory_path):
        list_files = os.listdir(directory_path)
        questions, answers = [], []
        last_type = None
        for file in list_files:
            with open(os.path.join(directory_path, file), "rb") as f:
                for line in f:
                    line = line.decode("utf8")
                    if line.strip():
                        sentence = self.get_tokenizer(line[5:])
                        if line[0] == "1":
                            if line[0] != last_type:
                                questions.append(sentence)
                            else:
                                questions[-1] += ' ' + sentence
                        elif line[0] == "0":
                            if line[0] != last_type:
                                answers.append(sentence)
                            else:
                                answers[-1] += ' ' + sentence
                        else:
                            import pdb; pdb.set_trace()
                        last_type = line[0]

        return questions, answers

    def build_vocabulary(self, data):
        words_list = dict()
        for sentence in data:
            for word in sentence.split(" "):
                if word in words_list:
                    words_list[word] += 1
                else:
                    words_list[word] = 1

        return words_list

    def filter_data(self, questions, answers):
        filted_question = []
        filted_answer = []
        for index, question in enumerate(questions):
            answer = answers[index]
            len_question = len(question)
            len_answer = len(answer)
            if (len_question <= self.limit['maxq']) and (len_question >= self.limit['minq']):
                if (len_answer <= self.limit['maxa']) and (len_answer >= self.limit['mina']):
                    filted_question.append(question)
                    filted_answer.append(answer)
        return filted_question, filted_answer

    def zero_pad(self, qtokenized, atokenized, w2idx):
        data_len = len(qtokenized)

        idx_q = np.zeros([data_len, self.limit['maxq']], dtype=np.int32)
        idx_a = np.zeros([data_len, self.limit['maxa']], dtype=np.int32)

        for i in range(data_len):
            q_indices = self.pad_seq(qtokenized[i], w2idx, self.limit['maxq'])
            a_indices = self.pad_seq(atokenized[i], w2idx, self.limit['maxa'])

            idx_q[i] = np.array(q_indices)
            idx_a[i] = np.array(a_indices)

        return idx_q, idx_a


    def pad_seq(self, seq, lookup, maxlen):
        indices = []
        for word in seq:
            if word in lookup:
                indices.append(lookup[word])
            else:
                indices.append(lookup[self.UNK])
        return indices + [0]*(maxlen - len(seq))

    def process_data(self, directory_path):
        print 'Start read data'
        questions, answers = self.get_data_from_files(directory_path)

        print 'Start filter data'
        filted_q, filted_a = self.filter_data(questions, answers)
        data = filted_q + filted_a

        print 'Start segment sentence to words'
        qtokenized = [ wordlist.split(' ') for wordlist in filted_q ]
        atokenized = [ wordlist.split(' ') for wordlist in filted_a ]

        print 'Start build vocabulary'
        word_list = self.build_vocabulary(data)
        word_array = []
        for word, number_occurrences in word_list.iteritems():
            word_array.append(word)
        print len(word_list)
        print len(data)

        index2word = ['_'] + [self.UNK] + word_array
        word2index = dict([(word, index) for index, word in enumerate(index2word)])

        print 'Start zero padding'
        idx_q, idx_a = self.zero_pad(qtokenized, atokenized, word2index)

        print 'Start save numpy arrays to disk'
        np.save('idx_q.npy', idx_q)
        np.save('idx_a.npy', idx_a)

        metadata = {
            'w2idx' : word2index,
            'idx2w' : index2word,
            'limit' : self.limit
        }

        with open('metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

    def load_data(self, path=''):
        with open(path + 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        idx_q = np.load(path + 'idx_q.npy')
        idx_a = np.load(path + 'idx_a.npy')
        return metadata, idx_q, idx_a

if __name__ == '__main__':
    data = Data()
    source_directory_path = "./data/conversations"
    data.process_data(source_directory_path)
