#encoding=utf-8
import re
import gensim
from gensim.models import KeyedVectors
#from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import string
import json
#import jieba
import time
import codecs
import msgpack


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


def load_wv_vocab(embed_file):
    '''Load tokens from word vector file.

    Only tokens are loaded. Vectors are not loaded at this time for space efficiency.

    Args:
        file (str): path of pretrained word vector file.

    Returns:
        set: a set of tokens (str) contained in the word vector file.
    '''
    vocab = []
    with open(embed_file) as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim]).strip() # a token may contain space
            if not token in vocab:
                 vocab.append(token)

    embs = np.zeros((len(vocab)+1, wv_dim))
    w2id = set([w for i, w in enumerate(vocab)])
    with open(embed_file) as f:
        for line in f:
            elems = line.split()
            #token = normalize_text(''.join(elems[0:-wv_dim]))
            token = ''.join(elems[0:-wv_dim]).strip()
            print(token)
            if token in w2id:
                #print(elems[-wv_dim:])
                embs[len(vocab) - len(w2id)] = [float(v) for v in elems[-wv_dim:]]
                w2id.remove(token)
    return vocab, embs

wv_dim = 100
wv_file = 'glove.6B.100d.txt'
#wv_vocab, embeddings = load_wv_vocab(wv_file)

#with open('vocab_embs.msgpack', 'wb') as f:
#       msgpack.dump({'vocab':wv_vocab, 'embs':embeddings.tolist()}, f)

with open('vocab_embs.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
        wv_vocab = data['vocab']
        embeddings = np.array(data['embs'])

starting = [['how', 'what', 'where', 'why', 'when', 'which'], ['on what', 'which one',  'in what'], ['are', 'is', 'were', 'was', 'do', 'does', '\'s' ]]
q_type = []
qtype_dict = {'ABBR':0, 'DESC':1, 'ENTY':2, 'HUM':3, 'LOC':4, 'NUM':5}

def load_data(filename):
    res = []
    with  codecs.open(filename, 'r',errors='ignore', encoding='utf-8') as f:
        for line in f:
            label, question = line.split(" ", 1)
            label = label[:label.index(':')]
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

            res.append((label, " ".join(tokens)))
            q_type.append(qtype_dict[label])
    return res

train_data = load_data('ques_class/train_5500.label')
test_data = load_data('ques_class/TREC_10.label')

MAX_TOKENS = len(wv_vocab)
embedding_dim = embeddings.shape[1]
hidden_dim_1 = 200
hidden_dim_2 = 100
NUM_CLASSES = 9
MAX_SEQUENCE_LENGTH = 50
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

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together)

pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic)

output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn)

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])


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



doc_train = doc_as_array[:len(train_data)]
left_train = left_context_as_array[:len(train_data)]
right_train = right_context_as_array[:len(train_data)]
target_train = target[:len(train_data)]

doc_val = doc_as_array[-len(test_data):]
left_val = left_context_as_array[-len(test_data):]
right_val = right_context_as_array[-len(test_data):]
target_val = target[-len(test_data):]

timestr = time.strftime("%Y%m%d-%H%M%S")
STAMP = 'lstm_' + timestr

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([doc_train, left_train, right_train], target_train, validation_data=([doc_val, left_val, right_val], target_val), epochs = 50, batch_size=512, shuffle=True, callbacks=[early_stopping, model_checkpoint])


model.load_weights(bst_model_path)
model.save_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
