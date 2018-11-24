# -*- coding:utf8 -*-
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import text_helpers
from nltk.corpus import stopwords

import os
import tensorflow as tf

freq_table_path = 'movie_vocab_freq.pkl'
word_dict_path = 'movie_vocab.pkl'
ckp_name = 'cbow_movie_embeddings.ckpt'
data_folder_name = '../temp'
embedding_size = 200
batch_size = 512
hidden_size1 = 128
sess = tf.Session()


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


def get_word_frequency(word_text, looktable):
    if word_text in looktable:
        return looktable[word_text]
    else:
        return 1.0/200000


def sentence_to_vec(sentence_list, embedding_size, looktable, word_table, embedding, a=1e-3 ):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence.split(' '))
        for word in sentence.split(' '):
            a_value = a / (a + get_word_frequency(word, looktable))  # smooth inverse frequency, SIF
            try:
                word_ix = word_table[word]
            except KeyError:
                word_ix = 0
            vs = np.add(vs, np.multiply(a_value, embedding[word_ix]))
            # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    print('生成sentence—vec完成')
    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    # print(sentence_vecs[-1])
    return np.array(sentence_vecs)


def sentence_to_vec_tfidf(sentence_list, embedding_size, looktable, word_table, embedding, a=1e-3):
    tfidf = TfidfVectorizer().fit(sentence_list)
    tfidf_m = tfidf.transform(sentence_list).todense()
    sentence_set = []
    list_len = len(sentence_list)
    for ix, sentence in enumerate(sentence_list):
        vs = np.zeros(embedding_size)
        for iw, word in enumerate(sentence.split(' ')):
            try:
                word_ix = word_table[word]
            except KeyError:
                word_ix = 0
            vs = np.add(vs, np.multiply(tfidf_m[ix, iw], embedding[word_ix]))
        sentence_set.append(vs)
    print('生成sentence—vec完成')
    return np.array(sentence_set)


def sentence_to_vec_avg(sentence_list, embedding_size, looktable, word_table, embedding, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence.split(' '))
        for word in sentence.split(' '):
            try:
                word_ix = word_table[word]
            except KeyError:
                word_ix = 0
            vs = np.add(vs, embedding[word_ix])
            # vs += sif * word_vector
        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    print('生成sentence—vec完成')
    return sentence_set


# 词典词频嵌入载入
with open(os.path.join(data_folder_name, word_dict_path), 'rb') as f:
    word_dict = pkl.load(f)
print('完成词字典载入')
with open(os.path.join(data_folder_name, freq_table_path), 'rb') as f:
    word_dict_freq = pkl.load(f)
print('完成词频字典载入')
embeddings = tf.Variable(tf.random_uniform([len(word_dict), embedding_size], -1, 1))
saver = tf.train.Saver({'embeddings': embeddings})
ckp_path = os.path.join(data_folder_name, ckp_name)
saver.restore(sess, ckp_path)
embed = sess.run(embeddings)
print('完成词嵌入载入')
# 载入句子
stops = stopwords.words('english')
texts, target = text_helpers.load_movie_data(data_folder_name)
texts = text_helpers.normalize_text(texts, stops)
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]
print('完成文本载入')

# Need to keep the indices sorted to keep track of document index
sentence_vec = sentence_to_vec_avg(texts, embedding_size, word_dict_freq, word_dict, embed)
train_indices = np.sort(np.random.choice(len(target), round(0.8 * len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
sentence_vec_train = np.array([x for ix, x in enumerate(sentence_vec) if ix in train_indices])
sentence_vec_test = np.array([x for ix, x in enumerate(sentence_vec) if ix in test_indices])
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
print('完成shuffle')

x_input = tf.placeholder(shape=[None, embedding_size], dtype=tf.float32)
y_output = tf.placeholder(shape=[None, 1], dtype=tf.int32)
W1 = tf.Variable(tf.truncated_normal([embedding_size, hidden_size1], stddev=0.05))
b1 = tf.Variable(tf.zeros([1, hidden_size1]))
# W0 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2], stddev=0.05))
# b0 = tf.Variable(tf.zeros([1, hidden_size2]))
W2 = tf.Variable(tf.truncated_normal([hidden_size1, 1], stddev=0.05))
b2 = tf.Variable(tf.zeros([1, 1]))
layer1 = tf.nn.relu(tf.add(tf.matmul(x_input, W1), b1))
# layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W0), b0))
model_output = tf.add(tf.matmul(layer1, W2), b2)
# y*log(sigmoid(model_output))+(1-y)*log(1-sigmoid(model_output))
logistic_loss = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.cast(y_output, tf.float32)))
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(y_output, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(logistic_loss, var_list=[W1, b1, W2, b2])
init = tf.global_variables_initializer()
sess.run(init)
print('Starting Logistic Doc2Vec Model Training')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(len(sentence_vec_train), size=batch_size)
    rand_x = sentence_vec_train[rand_index]
    # rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])
    feed_dict = {x_input: rand_x, y_output: rand_y}
    sess.run(train_step, feed_dict=feed_dict)
    if (i + 1) % 500 == 0:
        rand_index_test = np.random.choice(len(sentence_vec_test), size=len(target_test))
        rand_x_test = sentence_vec_test[rand_index_test]
        # Append review index at the end of text data
        rand_y_test = np.transpose([target_test[rand_index_test]])

        test_feed_dict = {x_input: rand_x_test, y_output: rand_y_test}

        i_data.append(i + 1)

        train_loss_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(logistic_loss, feed_dict=test_feed_dict)
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict=feed_dict)
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict=test_feed_dict)
        test_acc.append(test_acc_temp)
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))
