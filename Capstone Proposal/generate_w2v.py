import numpy as np
import gensim
from gensim.models import word2vec
from sklearn.decomposition import PCA
import logging
from pprint import pprint
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def generate_w2v():
    #with open('../20news-bydata/text8', 'r') as fp:
        #model = word2vec.Word2Vec([line.split(' ') for line in fp], size=200, min_count=0, window=5, workers=4)
    sentences = word2vec.Text8Corpus('../20news-bydata/text8')
    model = word2vec.Word2Vec(sentences, size=300)
    model.save('../20news-bydata/model_300/w2v.model')
    model.wv.save_word2vec_format('../20news-bydata/model_300/w2v.format')
    return model

def addtext(ax, x, y, text, props):
    ax.text(x, y, text, props, rotation=0)
    ax.text(x, y, text, props, rotation=0)
    ax.text(x, y, text, props, rotation=0)
    ax.text(x, y, text, props, rotation=225)
    ax.text(x, y, text, props, rotation=-45)

def w2v_visualization():
    model = gensim.models.Word2Vec.load('../20news-bydata/model_300/w2v.model')
    print (model.most_similar('college', topn=10))

    word_list = np.asarray(list(model.wv.vocab.keys()))
    print (word_list)
    w2v_word_list = [model[i] for i in word_list]
    clf = PCA(n_components=2)
    clf.fit(w2v_word_list)
    features = np.asarray(clf.transform(w2v_word_list))
    max_x = max([i[0] for i in features])
    min_x = min([i[0] for i in features])
    max_y = max([i[1] for i in features])
    min_y = min([i[1] for i in features])

    #word_list = ['school', 'college', 'university', 'boy', 'girl', 'google', 'amazon', 'company', 'car', 'bus']
    w2v_word_list = [model[i] for i in word_list]
    features = np.asarray(clf.transform(w2v_word_list))
    feature_len = len(features)

    fig, axs = plt.subplots(1, 1)
    for i ,label in enumerate(features):
        axs.scatter(label[0], label[1])
        axs.annotate(word_list[i], (label[0], label[1]))
    '''
    mask = np.random.random_integers(0, feature_len, 20)
    for label , word in zip(features[mask], word_list[mask]):
        axs.scatter(label[0], label[1])
        axs.annotate(word, (label[0], label[1]))
    '''
    axs.set_xticks(np.arange(min_x, max_x, 1))
    axs.set_yticks(np.arange(min_y, max_y, 1))
    plt.show()
    ### 取出词向量的矩阵
    ### 进行降维
    ### 抽取一部分词做可视化处理

if __name__ == '__main__':
    model = generate_w2v()
    #w2v_visualization()
