毕业项目说明
项目使用的开发语言为Python主要依赖项有：
1. 工具包 sklearn、tensorflow、keras
2. 数据集 20 Newsgroups,下载链接为http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
3. 训练好的300维glove词向量，下载地址为http://nlp.stanford.edu/data/glove.6B.zip

项目目录和文件说明:
./report.ipynb   项目主流程执行文件,可直接使用notebook执行
./keras_clf.py   深度学习分类器的实现流程
./helper.py      一些需要用到的辅助接口实现
./preprocess.py  预处理文件，主要实现加载数据并进行预处理等
./cnn_log.txt    使用keras_clf实现模型验证的日志文件

./data/20_newsgroups/ 存放数据集，全路径可表示为 ./data/20_newsgroups/alt.atheism/49960
./data/glove.6B/      存放预先训练好的词向量，全路径可表示为 ./data/glove.6B/glove.6B.300d.txt

项目运行过程和实现：
可以直接打开report.ipynb从上往下执行，执行时间约为半个小时左右。主要耗时在模型调参。
深度学习可以单独执行keras_clf.py这个文件,耗时较长约3.5小时左右。
