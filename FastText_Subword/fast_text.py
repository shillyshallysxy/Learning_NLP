import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
import text_helpers
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()
# Set Random Seeds
tf.set_random_seed(42)
np.random.seed(42)

# os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Make a saving directory if it doesn't exist
data_folder_name = 'G:\\python\DeepLearning\learn_tensorflow\part7\\temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

sess = tf.Session()

# Declare model parameters
tri_gram = 3
batch_size = 512
embedding_size = 200
# vocabulary_size = 7500
generations = 2000
model_learning_rate = 0.001

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 3       # How many words to consider left and right.

# Load the movie review data
stops = stopwords.words('english')
texts, target = text_helpers.load_movie_data(data_folder_name)
# Normalize text
print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)
# Texts must contain at least 3 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]
print('Creating Dictionary')
word_dictionary, max_length = text_helpers.build_dictionary(texts)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

word_dictionary_fast = text_helpers.build_dictionary_fast(texts)
vocabulary_size = len(word_dictionary_fast.keys())
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                              stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
x_inputs = tf.placeholder(tf.int32, shape=[batch_size, max_length])
y_target = tf.placeholder(tf.int32, shape=[batch_size, max_length])
embed_x = tf.zeros([batch_size, embedding_size])
embed_y = tf.zeros([batch_size, embedding_size])
for element in range(max_length):
    embed_x += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])
for element in range(max_length):
    embed_y += tf.nn.embedding_lookup(embeddings, y_target[:, element])

cosine = tf.reduce_sum(tf.multiply(embed_x, embed_y), 1)
loss = tf.reduce_mean(-tf.log_sigmoid(cosine))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size, window_size)
    batch_inputs = text_helpers.word2ngrim(batch_inputs, word_dictionary_rev, word_dictionary_fast, max_length)
    batch_labels = text_helpers.word2ngrim(batch_labels, word_dictionary_rev, word_dictionary_fast, max_length)

    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    # if i == 173:
    #     print(type(batch_inputs),i)
    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)
    if (i + 1) % 100 == 0:
        # print(sess.run(cosine, feed_dict=feed_dict))
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))
    if (i + 1) % 500 == 0:
        print(sess.run(cosine, feed_dict=feed_dict))
embed_m = sess.run(embeddings)
grim_word = text_helpers.word2ngrim(word_dictionary.keys(), word_dictionary_rev, word_dictionary_fast, max_length)
grim_word_embed = np.reshape([embed_m[x] for b_x in grim_word for x in b_x], newshape=[-1, max_length, embedding_size])
embed_final = np.squeeze(np.sum(grim_word_embed, axis=1))
U, s, Vh = np.linalg.svd(embed_m, full_matrices=False)
x_max = 0
y_max = 0
print('s_matrix: ', s)
for ix, word in enumerate(word_dictionary.keys()):
    if ix<1000:
        plt.text(U[ix, 0], U[ix, 1], word)
        if U[ix, 0] > x_max:
            x_max = U[ix, 0]
        if U[ix, 1] > y_max:
            y_max = U[ix, 1]
plt.ylim([0, y_max])
plt.xlim([0, x_max])
plt.show()