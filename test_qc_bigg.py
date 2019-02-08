encoding=utf-8
import codecs
import re
import gensim
from gensim.models import KeyedVectors
#from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import string
import json
#import jieba
import time

import msgpack
from collections import Counter

from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

import unicodedata


def normalize_text(text):
    return unicodedata.normalize('NFD', text)

with open('vocab_embs_300.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
        wv_vocab = data['vocab']
        embeddings = np.array(data['embs'])

q_dict = {'ABBR':0, 'ENTY':1, 'HUM':2, 'LOC':3, 'NUM':4}
#q_dict = {'ABBR':0, 'DESC':1, 'ENTY':2, 'HUM':3, 'LOC':4, 'NUM':5}

question_labels = [['What is the city in which Maurizio Pellegrin lives called', 'LOC'], ['What is a fear of food', 'ENTY'], ['the city in which Maurizio Pellegrin lives called', 'LOC'], ['a fear of food', 'ENTY'], ['Maria buys coffee', 'HUM'],['it happened in July', 'ENTY'], ['around 453 times', 'NUM'] ]

with open('ints_nonover_ni3_2.msgpack', 'rb') as f:
    ints = msgpack.load(f, encoding='utf8')

q_en = []
q_type = []
for line in question_labels:
    q_en.append(line[0])
    q_type.append(q_dict[line[1]])


MAX_TOKENS = len(wv_vocab)
embedding_dim = embeddings.shape[1]
hidden_dim_1 = 250
hidden_dim_2 = 150
NUM_CLASSES = len(q_dict)
hidden_dim_3 = 100
MAX_SEQUENCE_LENGTH = 34
VALIDATION_SPLIT = 0.1

document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

embedder = Embedding(MAX_TOKENS + 1, embedding_dim, weights = [embeddings], trainable = False)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)

forward = LSTM(hidden_dim_1, return_sequences = True)(l_embedding)
backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding)
together = concatenate([forward, doc_embedding, backward], axis = 2)

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "relu"))(together)
semantic = TimeDistributed(Dense(hidden_dim_3, activation = "relu"))(semantic)

pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_3, ))(semantic)

output = Dense(NUM_CLASSES, input_dim = hidden_dim_3, activation = "softmax")(pool_rnn)

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])

model.load_weights('lstm_bigg_20190129-213055.h5')

doc_as_array = []
left_context_as_array = []
right_context_as_array = []

w2id = {w: i for i, w in enumerate(wv_vocab)}
idx2w = {i:a for a,i in q_dict.items()}

for a, vals in ints.items():
        doc_as_array = []
        left_context_as_array = []
        right_context_as_array = []
        for text in vals:
            text = text.strip().lower()
            text = re.sub('\s+', ' ', text)
            tokens = text.split(' ')
            tokens = [w2id[token] if token in wv_vocab else MAX_TOKENS for token in tokens]

            doc_as_array.append(np.array([tokens])[0])

            left_context_as_array.append(np.array([[MAX_TOKENS] + tokens[:-1]])[0])

            right_context_as_array.append(np.array([tokens[1:] + [MAX_TOKENS]])[0])

        doc = pad_sequences(doc_as_array, maxlen=MAX_SEQUENCE_LENGTH)
        left = pad_sequences(left_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
        right = pad_sequences(right_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
        target = to_categorical(q_type, num_classes=NUM_CLASSES)

        counts = [idx2w[w] for w in backend.eval(backend.argmax(model.predict([doc, left, right]), axis=1))]
        print('a=%d'%a, Counter(counts))
"""
for text in q_en:
    text = text.strip().lower()
    text = re.sub('\s+', ' ', text)
    tokens = text.split(' ')
    tokens = [w2id[token] if token in wv_vocab else MAX_TOKENS for token in tokens]

    doc_as_array.append(np.array([tokens])[0])
   
    left_context_as_array.append(np.array([[MAX_TOKENS] + tokens[:-1]])[0])

    right_context_as_array.append(np.array([tokens[1:] + [MAX_TOKENS]])[0])



doc_as_array = pad_sequences(doc_as_array, maxlen=MAX_SEQUENCE_LENGTH)
left_context_as_array = pad_sequences(left_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
right_context_as_array = pad_sequences(right_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
target = to_categorical(q_type, num_classes=NUM_CLASSES)


doc = doc_as_array
left = left_context_as_array
right = right_context_as_array
target = target

idx2w = {i:a for a,i in q_dict.items()}

for i in range(len(doc)):
        print(idx2w[backend.eval(backend.argmax(model.predict([doc[i], left[i], right[i]]), axis=1))[0]])
        print(backend.eval(backend.argmax(model.predict([doc[i], left[i], right[i]]), axis=1))[0])
        print(question_labels[i])
"""

starting = [['how', 'what', 'where', 'why', 'when', 'which'], ['on what', 'which one',  'in what'], ['are', 'is', 'were', 'was', 'do', 'does', '\'s', 'did' ]]
qtype_dict = {'ABBR':0, 'ENTY':1, 'HUM':2, 'LOC':3, 'NUM':4}
#qtype_dict = {'ABBR':0, 'DESC':1, 'ENTY':2, 'HUM':3, 'LOC':4, 'NUM':5}
q_type = []

def load_data(filename):
    global max_seq
    res = []
    with  codecs.open(filename, 'r',errors='ignore', encoding='utf-8') as f:
        for line in f:
            label, question = line.split(" ", 1)
            label = label[:label.index(':')]
            if label.lower() == 'desc':
                  continue
            question = question.strip()[:-2].strip().lower()
            text = re.sub('\s+', ' ', question)
            tokens = text.split(' ')
            if tokens[0] in starting[0]:
                  if " ".join(tokens[:2]) in starting[1]:
                       tokens = tokens[2:]
                  else:
                       tokens = tokens[1:]
            elif " ".join(tokens[:2]) in starting[1]:
                 tokens = tokens[2:]
            if tokens[0] in starting[2]:
                 tokens = tokens[1:]
            if len(tokens)>max_seq:
                 max_seq = len(tokens)
            #print(question, " ".join(tokens))
            res.append((label, " ".join(tokens)))
            q_type.append(qtype_dict[label])
    return res

max_seq = 0
train_data = load_data('ques_class/train_5500.label')
test_data = load_data('ques_class/TREC_10.label')


doc_as_array = []
left_context_as_array = []
right_context_as_array = []

w2id = {w: i for i, w in enumerate(wv_vocab)}

for text in train_data+test_data:
    text = text[1].strip().lower()
    text = re.sub('\s+', ' ', text)
    tokens = text.split(' ')
    tokens = [w2id[token] if token in wv_vocab else MAX_TOKENS for token in tokens]

    doc_as_array.append(np.array([tokens])[0])

    left_context_as_array.append(np.array([[MAX_TOKENS] + tokens[:-1]])[0])

    right_context_as_array.append(np.array([tokens[1:] + [MAX_TOKENS]])[0])



doc_as_array = pad_sequences(doc_as_array, maxlen=MAX_SEQUENCE_LENGTH)
left_context_as_array = pad_sequences(left_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
right_context_as_array = pad_sequences(right_context_as_array, maxlen=MAX_SEQUENCE_LENGTH)
target = to_categorical(q_type, num_classes=NUM_CLASSES)


doc_val = doc_as_array[-len(test_data):]
left_val = left_context_as_array[-len(test_data):]
right_val = right_context_as_array[-len(test_data):]
target_val = target[-len(test_data):]


print(model.evaluate([doc_val, left_val, right_val], target_val))

print(Counter(backend.eval(backend.argmax(model.predict([doc_val, left_val, right_val]), axis=1))))
print(q_dict)
