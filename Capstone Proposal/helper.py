import os
from time import time
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

def to_sequence_by_w2v(sentence):
    sentence = [sentence]
    MAX_WORDS = 100
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(sentence)
    sequence_train = tokenizer.texts_to_sequences(sentence)
    word_index = tokenizer.word_index

    embeddings_index = {}
    with open(os.path.join('../data/glove.6B', 'glove.6B.100d.txt')) as fp:
        for line in fp:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros([len(word_index) + 1, 100])
    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_tensor = tf.Variable(embedding_matrix, dtype='float32')
    X = tf.placeholder('int32', [None, len(sequence_train[0])], name='input')
    embedding_X = tf.nn.embedding_lookup(embedding_tensor, X)
    init = tf.global_variables_initializer()

    seq = None
    with tf.Session() as sess:
        sess.run(init)
        seq = sess.run(embedding_X, feed_dict={X: sequence_train})
    return seq[0]


def get_doc_word_freq(data):
    vectorizer = CountVectorizer()
    t = time()
    data_array = vectorizer.fit_transform(data)
    #print ('step 1 : ', time() - t)
    
    t = time()
    word_list = [word for word,index in sorted(vectorizer.vocabulary_.items(), key=itemgetter(1))]  
    #print ('step 2 : ', time() - t)
    
    t = time()
    total_word_count = len(word_list)
    words_counter_dict = dict()
    data_array = data_array.toarray()
    for i in range(0, total_word_count):
        words_counter_dict[word_list[i]] = data_array[:,i].sum()
    #print ('step 3 : ', time() - t)
    
    t = time()
    doc_words_counter_list = [data_array[j].sum() for j in range(0, data_array.shape[0])]
    #print ('step 4 : ', time() - t)
    
    return doc_words_counter_list, words_counter_dict
