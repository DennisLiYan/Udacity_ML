import os
import numpy as np
from preprocess import load_data

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding,Dropout
from keras.models import Model

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1500

BASE_DIR = '../data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')

category = None
data = load_data('../data/20_newsgroups', categories = category, shuffle=True, skip_header=False)
X_data, X_test, Y_data, Y_test = \
    train_test_split(data['data'], data['labels'], test_size = 0.1, random_state=42)
X_train, X_develop, Y_train, Y_develop = train_test_split(X_data, Y_data, test_size = 0.2, random_state=42)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
word_index = tokenizer.word_index

train_seq = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
train_labels = np.eye(len(data['label_names']))[Y_train]

print ('embeddings_index ...')
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')) as fp:
    for line in fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

EMBEDDING_DIM = 300
num_words = min(len(word_index), MAX_NUM_WORDS)
embedding_matrix = np.zeros([num_words, EMBEDDING_DIM])
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

def init_model_graph():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(20, activation='softmax')(x)
    model = Model(sequence_input, preds)
    return model

model = init_model_graph()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

sequences = tokenizer.texts_to_sequences(X_develop)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(Y_develop))

model.fit(train_seq,
          train_labels,
          batch_size=256,
          epochs=20,
          validation_data=(data, labels))
model.save('./model.ckpt')

model.load_weights('./model.ckpt')
print (model.evaluate(data, labels, batch_size = 256))