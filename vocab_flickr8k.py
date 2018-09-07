# loadbars to track the run/speed
from tqdm import tqdm, trange
# numpy for arrays/matrices/mathematical stuff
import numpy as np
# nltk for tokenizer
from nltk.tokenize import wordpunct_tokenize
import os
import pickle
from collections import Counter
from collections import defaultdict
from datetime import datetime
from flickr8k_data_processor import *

class DataProcessor():
    def __init__(self, annotations, train_data, vocab_size = 30000, filename=None, pad='<PAD>',start='<START>',end='<END>',unk='<UNK>'):
        self.vocab_size = vocab_size
        self.PAD = pad
        self.START = start
        self.END = end
        self.UNK = unk
        if filename == None:
            filename = 'flickr8k_vocab_'+str(self.vocab_size)+'.pkl'
        self.filename = filename
        if os.path.isfile(self.filename):
            self.vocab, self.vocab_size, self.vocab_weight = self.load()
        else:
            self.vocab, self.vocab_size, self.vocab_weight = self.build_vocab(annotations, train_data)
        self.w2i, self.i2w = self.build_dicts()

    def build_dicts(self):
        """
        creates lookup tables to find the index given the word
        and the otherway around
        """
        w2i = defaultdict(lambda: w2i[self.UNK])
        i2w = dict()
        for i,w in enumerate(self.vocab):
            i2w[i] = w
            w2i[w] = i
        return w2i, i2w

    def build_vocab(self, annotations, train_data):
        """
        builds a vocabulary with the most occuring words, in addition to
        the UNK token at index 1 and PAD token at index 0.
        START and END tokens are added to the vocabulary through the
        preprocessed sentences.
        with vocab size none, all existing words in the data are used
        """
        vocab = Counter()
        for line in annotations:
            sp = line.split('\t')
            if sp[0][:-2] in train_data:
                sent = sp[1].split()
                for w in sent:
                    vocab[w] += 1

        vocab = [k for k,_ in vocab.most_common(self.vocab_size - 4)] #minus 4 because of the default tokens
        vocab_weights = list(range(len(vocab)))
        vocab = [self.PAD,self.UNK,self.START,self.END] + vocab # padding needs to be first, because of the math
        vocab_weights = [0.,1.,1.,1.] + vocab_weights
        return vocab,len(vocab), vocab_weights

    def save(self):
        pickle.dump(self.vocab, open(self.filename, 'wb'))

    def load(self):
        vocab = pickle.load(open(self.filename, 'rb'))
        vocab_size = len(vocab)
        vocab_weights = [0.,1.,1.,1.] + list(range(len(vocab)-4))
        return vocab, vocab_size, vocab_weights
