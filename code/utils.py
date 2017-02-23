from nltk.corpus import stopwords
import sys
import tensorflow as tf
import string
import os
import time
import random
import numpy as np
import cPickle as pickle

# A basic punctutation filter with "_" removed because 
# we use "_" to delimit spaces in an n-gram
def base_filter():
    f = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
    f = f.replace("'", '')
    f += '\t\n'
    return f

table = string.maketrans(base_filter(), " "*len(base_filter()))

my_stopwords = ["hi", "also", "berkeley", "edu", "http", "one", "two", "hello", "three"]


# Processes a document by splitting it up into a list of words
# (or n-grams according to the arg bigram) and filtering out
def bigram_text_to_word_sequence(text, bigram, filters=base_filter(), lower=False, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = str(text).translate(table)
    seq = text.split(split)
    sentences = [_f for _f in seq if _f]
    return bigram(sentences)
 

# Does the same as above, but then returns a hash-based non-unique
# one-hot-encoding of the word sequences
def bigram_one_hot(text, n, bigram, filters=base_filter(), lower=False, split=" "):
    seq = bigram_text_to_word_sequence(text, bigram, filters=filters, lower=lower, split=split)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]


def one_hot(seq, n):
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]

# I found it nice to functionize checking for stopwords
# This would also allow us to easily filter for more advanced
# stopping conditions like POS-type. 
def is_not_stopword(s, tagger):
    if len(s) == 0 or s.lower() in my_stopwords:
        return False
    if len(s.split('_')) == 1:
        s = str(s)
        if s.lower() in stopwords.words('english') or s.lower().split("'")[0] in stopwords.words('english'):
            return False 
        try:
            int(s)
            return False
        except:
            pass
        if tagger.tag([s])[0][1] in ["NUM", "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "CD", "MD", "VBG", "VBN", "VGP", "VBD", "VBP", "VBZ", "VB", "IN"]:
            return True 
        if len(s) < 2:
            return False
        return True
    else:
        s_s = [is_not_stopword(k, tagger) for k in s.split('_')]
        return False not in s_s

# Parse the 333,000 most frequent english words
def parse_frequency():
    D = {}
    with open("../datasets/count_1w.txt", "rb") as f:
        lines = f.readlines()
        for line in lines:
            word, count = line.strip().replace("\t", " ").split(" ")
            D[word] = float(count)
    return D


# Nice method from stack overflow for line re-writing
def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def clean_print(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# This is an additional metric that is needed in my implementation
# Ideally should allow for a signifigant training-time speed-up
def per_sample_loss_metric(ytrue, ypred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(ypred, tf.reshape(tf.cast(ytrue, tf.int32), [-1]))

class Timer:

    def __init__(self):
        self.silent = False

    def tic(self, msg = None): 
        self.t = time.time()
        self.ticked = True
        self.msg = msg

    def toc(self):
        if not self.ticked:
            raise Exception()
        if not self.silent:
            if not self.msg:
                print("Time elapsed: {0}".format(time.time() - self.t))
            else:
                print("Time elapsed since {0}: {1}".format(self.msg, time.time() - self.t))
        self.ticked = False
        self.msg = None

    def silence(self):
        self.silent = not self.silent

T = Timer()

def data_generator(PATH, BATCH_SIZE):
    j = 0
    T2 = Timer()
    T2.silence()
    while True:
        T2.tic("Loading data")
        with open(PATH, "rb") as f:
            data_labels_words = pickle.load(f)[0]
        T2.toc()
        i = 0
        have_wrapped = False
        T2.tic("Shuffling")
        order = range(len(data_labels_words))
        random.shuffle(order)
        T2.toc()
        while i < len(data_labels_words) and not have_wrapped:
            toYield_x = np.zeros((BATCH_SIZE))
            toYield_y = np.zeros((BATCH_SIZE))
            T2.tic("Iterating")
            for _iter in range(BATCH_SIZE):
                toYield_x[_iter], toYield_y[_iter] = data_labels_words[order[i]][0], data_labels_words[order[i]][1]
                i += 1 
                if i >= len(data_labels_words):
                    have_wrapped = True
                    i = 0
            T2.toc()
            yield toYield_x, toYield_y[:, np.newaxis]
        j += 1
        if not os.path.isfile(PATH + "CBOW_{0}.p".format(j)):
            j = 0
        restart_line()
        clean_print("Finished full data epoch.")
        print("")