import csv
import numpy as np
import random
import re
import nltk

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import codecs
import random

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from tqdm import tqdm 
from torch.autograd import Variable

#--------------------------- TRAINING -----------------------------------------
def train_model(model, optimizer, loss_fn, feature_train, target_train, feature_valid, target_valid, feature_test, target_test):
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        predictions = model(feature_train).squeeze(1)
        loss = loss_fn(predictions, target_train)
        acc = accuracy(predictions, target_train)
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        epoch_acc = acc
  
        model.eval()
  
        with torch.no_grad():

            predictions_valid = model(feature_valid).squeeze(1)
            loss = loss_fn(predictions_valid, target_valid)
            acc = accuracy(predictions_valid, target_valid)
            valid_loss = loss.item()
            valid_acc = acc

        print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
    
    model.eval()

    with torch.no_grad():
 
        predictions = model(feature_test).squeeze(1)
        u = torch.FloatTensor(target_test)
        loss = loss_fn(predictions, u)
        acc = accuracy(predictions, u)
        print(f'| Test Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')
        f_measure(predictions, target_test)

    print(np.count_nonzero(target_test))

#--------------------------- ACCURACY -----------------------------------------
def accuracy(output, target):
    # output = torch.round(torch.sigmoid(output))
    output = torch.nn.functional.log_softmax(output).max(dim = 1)[1]
    correct = (output == target).float()
    acc = correct.sum() / len(correct) 
    return acc    

def f_measure(output, gold):  
    # pred = torch.round(torch.sigmoid(output))
    pred = torch.nn.functional.log_softmax(output).max(dim = 1)[1]
    pred = pred.detach().cpu().numpy()

    test_pos_preds = np.sum(pred)
    test_pos_real = np.sum(gold)

    true_positives = (np.logical_and(pred, gold)).astype(int)
    true_positives = np.sum(true_positives)
    print(true_positives)

    precision = true_positives / test_pos_preds
    recall = true_positives / test_pos_real

    fscore = 2.0 * precision * recall / (precision + recall)
    print("Test: Recall: %.2f, Precision: %.2f, F-measure: %.2f\n" % (recall, precision, fscore))  

#--------------------------- MODELS -------------------------------------------
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels, window_size, output_dim, dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # in_channels -- 1 text channel
        # out_channels -- the number of output channels
        # kernel_size is (window size x embedding dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(window_size,embedding_dim))

        # the dropout layer
        self.dropout = nn.Dropout(dropout)

        # the output layer
        self.fc = nn.Linear(out_channels, output_dim)        
        
    def forward(self, x):
        # (batch size, max sent length)
        embedded = self.embedding(x)
                
        # (batch size, max sent length, embedding dim)
        
        # images have 3 RGB channels 
        # for the text we add 1 channel
        embedded = embedded.unsqueeze(1)
        
        #(batch size, 1, max sent length, embedding dim)        
        feature_maps = self.conv(embedded)

        #Q. what is the shape of the convolution output ?
        feature_maps = feature_maps.squeeze(3)
        
        #Q. why do we reduce 1 dimention here ?                
        feature_maps = F.relu(feature_maps)
  
        #the max pooling layer
        pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])
        
        pooled = pooled.squeeze(2)
        #Q. what is the shape of the pooling output?
        dropped = self.dropout(pooled)
 
        preds = self.fc(dropped)        
        return preds


#--------------------------- TOKENIZERS ---------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
stop = stopwords.words("english")
ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

def tokenize_stemming(corpus):
    return get_tokenized_corpus(corpus, ps.stem)

def tokenize_lemmatization(corpus):
    return get_tokenized_corpus(corpus, lmtzr.lemmatize)

def tokenize(corpus):
    return get_tokenized_corpus(corpus, lambda x: x)

def get_tokenized_corpus(corpus, f, is_stop=False):
    tokenized_corpus = []
    for sentence in corpus:
        tokenized_sentence = []
        for token in sentence.split(' '): 
            if is_stop and token in stop:
                continue;
            tokenized_sentence.append(f(token))
        tokenized_corpus.append(tokenized_sentence)
    return tokenized_corpus

def get_vocabulary(tokenized_corpus):
    vocabulary = [] # Let us put all the tokens (mostly words) 
                    # appearing in the vocabulary in a list
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    return vocabulary

def get_word2idx(tokenized_corpus, vocabulary):  
    word2idx = {w: idx + 1 for (idx, w) in enumerate(vocabulary)}
    # we reserve the 0 index for the placeholder token
    word2idx['<pad>'] = 0
    return word2idx

def get_idx2word(vocabulary):
    return {idx: w for (idx, w) in enumerate(vocabulary)}

def parse_input(tokenized_corpus, word2idx, labels, max_len):
    # we index our sentences
    vectorized_sentences = [[word2idx[token] for token in sentence if token in word2idx] for sentence in tokenized_corpus]
  
  
    # we create a tensor of a fixed size filled with zeroes for padding
    sentences_tensor = Variable(torch.zeros((len(vectorized_sentences), max_len))).long()
    sentences_lengths = [len(sentence) for sentence in vectorized_sentences]

    # we fill it with our vectorized sentences 
    for idx, (sentence, sentence_len) in enumerate(zip(vectorized_sentences, sentences_lengths)):
        sentences_tensor[idx, :sentence_len] = torch.LongTensor(sentence)

    labels_tensor = torch.FloatTensor(labels)

    return sentences_tensor, labels_tensor

#--------------------------- PARSERS ------------------------------------------

def parse_dataset(filename):
    rows = read_csv(filename)
    train_corpus = [translate(row['tweet'].lower()).lower() for row in rows]
    train_labels_a = [1 if row['subtask_a'] == 'OFF' else 0 for row in rows]
    train_labels_b = [row['subtask_b'] for row in rows]
    train_labels_c = [row['subtask_c'] for row in rows]
    return train_corpus, train_labels_a, train_labels_b, train_labels_c

def parse_dataset_task_b(filename):
    rows = read_csv(filename)
    train = [row for row in rows if row['subtask_b'] != 'NULL']
    train_corpus = [translate(row['tweet'].lower()).lower() for row in train]
    train_labels_b = [0 if row['subtask_b'] == 'UNT' else 1 for row in train]
    return train_corpus, train_labels_b

def parse_dataset_task_c(filename):
    rows = read_csv(filename)
    train = [row for row in rows if row['subtask_c'] != 'NULL']
    train_corpus = [translate(row['tweet'].lower()).lower() for row in train]

    train_labels_c = []
    for row in train:
        if row['subtask_c'] == 'IND':
            train_labels_c.append(0)
        elif row['subtask_c'] == 'GRP':
            train_labels_c.append(1)
        else:
            train_labels_c.append(2)

    return train_corpus, train_labels_c


def parse_dataset_t_b(filename):
    rows = read_csv(filename)
    train = [row for row in rows if row['subtask_b'] != 'NULL']
    train_corpus = [keep_only_spaces(row['tweet'].lower()).lower() for row in train]
    train_labels_b = [0 if row['subtask_b'] == 'UNT' else 1 for row in train]
    return train_corpus, train_labels_b

def translate(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        fileName = "slang.txt"
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            _str = keep_only_spaces(_str)
            changed = False
            for row in dataFromFile:
                if _str.upper() == row[0]:
                    user_string[j] = row[1]
                    changed = True
            if not changed:
                user_string[j] = _str
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(user_string)


def read_csv(path):
    rows = []
    with open(path) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            rows.append(row)
    return rows

#--------------------------- DATA AUGMENTATION --------------------------------

def augment_negatives(corpus, labels):
    p, n = [], []
    for idx, label in enumerate(labels):
        if label == 0: 
            p.append(corpus[idx])
        else:
            n.append(corpus[idx])

    for negative in n:
        r = random.randint(0, len(p)) - 1
        
        even = random.randint(0, 5)
        
        new_negative = ""
        
        if even % 2 == 1:
            new_negative = negative + " " + p[r]
        else:
            new_negative = p[r] + " " + negative
            
        corpus.append(new_negative)
        labels.append(1)  
        
    return corpus, labels

#-------------------------- DATA PROCESSING -----------------------------------

def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop])    
    
def shuffle_corpus(train_corpus, train_labels_a):
    u = list(zip(train_corpus, train_labels_a))    
    random.shuffle(u)
    train_corpus, train_labels_a = zip(*u)


def keep_only_spaces(text):
    return re.sub(r'([^\s\w]|_)+', '', text)
    
if __name__ == '__main__':
    print("one.py is being run directly")

