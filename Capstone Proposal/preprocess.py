import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import re

from collections import Counter
from operator import attrgetter,itemgetter

#from nltk.stem import WordNetLemmatizer

# 按目录读取文件，并且将文件内容、标签、标签名字组织在合理的数据结构中
def load_data(container_path = '', categories = None,
              skip_header = False, shuffle = False,
              random_state=0):
    category_names = [f for f in sorted(os.listdir(container_path))
                       if os.path.isdir(os.path.join(container_path, f))]
    file_names = []
    label_names = category_names
    labels = []
    data = []

    if categories != None:
        category_names = [c for c in category_names if c in categories]

    for index, name in enumerate(category_names):
        cate_dir_name = os.path.join(container_path, name)
        cate_file_names = [os.path.join(cate_dir_name, c) for c in os.listdir(cate_dir_name)]
        labels.extend(len(cate_file_names) * [index])
        file_names.extend(cate_file_names)

    file_names = np.asarray(file_names)
    labels = np.asarray(labels)
    if shuffle:
        random_state = np.random.RandomState(random_state)
        indices = np.arange(file_names.shape[0])
        random_state.shuffle(indices)
        file_names = file_names[indices]
        labels = labels[indices]

    for f in file_names:
        with open(f, 'r', encoding='latin1') as fp:
            t = fp.read()
            if skip_header:
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
            data.append(t)
    return dict({
        'file_names': file_names,
        'label_names': label_names,
        'labels': labels,
        'data':data
    })

def test():
    #category = ['alt.atheism', 'comp.graphics']
    category = None
    train_data = load_data('../data/origin/20news-bydate-train', categories = category)
    test_data = load_data('../data/origin/20news-bydate-test', categories = category)

    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=0, max_df=0.5, ngram_range=(1, 1))
    tfidf.fit(train_data['data'])

    clf = LinearSVC(penalty="l2", dual=False, tol=1e-3, random_state=42, C=1)
    X_train_data = tfidf.transform(train_data['data'])
    clf.fit(X_train_data, train_data['labels'])
    preds = clf.predict(X_train_data)
    print(classification_report(train_data['labels'], preds, target_names=np.asarray(train_data['label_names'])))
    #print (accuracy_score(train_data['labels'], preds))

    X_test_data = tfidf.transform(test_data['data'])
    preds = clf.predict(X_test_data)
    print(classification_report(test_data['labels'], preds, target_names=np.asarray(test_data['label_names'])))
    #print (accuracy_score(test_data['labels'], preds))

def proprocess_document():
    #wnl = WordNetLemmatizer()
    #category = ['alt.atheism']
    category = None
    train_data = load_data('../data/origin/20news-bydate-train', categories = None)
    test_data = load_data('../data/origin/20news-bydate-test', categories = None)
    for index, doc in enumerate(train_data['data']):
        with open(train_data['file_names'][index], 'w', encoding='latin1') as fp:
            fp.write(' '.join([wnl.lemmatize(word.lower()) for word in doc.split(' ')]))

    for index, doc in enumerate(test_data['data']):
        with open(test_data['file_names'][index], 'w', encoding='latin1') as fp:
            fp.write(' '.join([wnl.lemmatize(word.lower()) for word in doc.split(' ')]))

import math

def explore_category_distribution():
    train_data = load_data('../data/origin/20news-bydate-train', categories = None)
    category_counter = dict()
    for index, name in enumerate(train_data['label_names']):
        labels = np.asarray(train_data['labels'])
        category_counter[name] = len(labels[labels == index])

    category_counter = np.asarray(sorted(category_counter.items(), key = itemgetter(1)))
    print (category_counter)
    index = [0,1,10,11,18,19]
    print (category_counter[0])
    fig, ax = plt.subplots(figsize=(60,20))
    ax.bar([i[0] for i in category_counter[index]], height = [int(i[1]) for i in category_counter[index]], color='navy')
    plt.show()

def explore_word_frequency():
    train_data = load_data('../data/origin/20news-bydate-train', categories = None)
    test_data = load_data('../data/origin/20news-bydate-test', categories = None)
    counter = Counter()
    total_lines = 0
    for d in train_data['data']:
        for index,line in enumerate(d.split('\n')):
            counter.update(re.findall('[A-Za-z-]*', line))
        total_lines += index
    print (counter.most_common(15))
    print ('total train lines %d' % (total_lines))
    print ('total train words %d' % (sum([c for w, c in counter.items()])))
    print ('total train article num is %d' % (len(train_data['data'])))
    print ('total test article num is %d' % (len(test_data['data'])))

    fig, ax = plt.subplots(figsize=(60,20))
    word_counter = sorted(counter.items(), key = itemgetter(1), reverse = False)[1:6]
    ax.bar([i[0] for i in word_counter], height = [int(i[1]) for i in word_counter], color='navy')
    plt.show()


def explore_features():
    train_data = load_data('../data/origin/20news-bydate-train', categories = None)
    data = train_data['data']
    r = Counter(data)
    print (r)

if __name__ == '__main__':
    proprocess_document()
    #test()
    #explore_features()
    #explore_category_distribution()
    #explore_word_frequency()
    #print (sum([j for w,j in Counter(['1', '2', '2']).items()]))
