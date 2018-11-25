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
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import numpy as np
from tensorflow.python.framework import ops

vocabulary_size = 253703
embedding_size = 200   # Word embedding size
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Make a saving directory if it doesn't exist
data_folder_name = 'temp'
sess = tf.Session()
with open(os.path.join(data_folder_name, 'movie_vocab.pkl'), 'rb') as f:
    word_dict = pickle.load(f)
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))
model_ckpt_path = os.path.join(data_folder_name, 'cbow_movie_embeddings.ckpt')
saver = tf.train.Saver({'embeddings': embeddings})
saver.restore(sess, model_ckpt_path)
embed_m = sess.run(embeddings)
U, s, Vh = np.linalg.svd(embed_m, full_matrices=False)
x_max = max(U[:, 0])
x_min = min(U[:, 0])
y_max = max(U[:, 1])
y_min = min(U[:, 1])
# print('s_matrix: ', s)
for ix, word in enumerate(word_dict.keys()):

    if ix < 1000:
        plt.text(U[ix, 0], U[ix, 1], word)
plt.ylim([y_min, y_max])
plt.xlim([x_min, x_max])
plt.show()
