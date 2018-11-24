import tensorflow as tf
import os
import csv
import requests
import io
from zipfile import ZipFile
import re
import numpy as np
import helper
import pickle
from nltk.corpus import stopwords

hidden_size = 10
max_sequence_length = 25
embedding_size = 200
batch_size = 250
min_word_frequency = 10
class_size = 2
learning_rate = 0.0005
epochs = 50

sess = tf.Session()

# Check if data was downloaded, otherwise download it and save for future use
data_folder_name = 'temp'
# tfrecord_name = 'movie_text2num.tfrecord'
tfrecord_train_name = 'movie_text2num_train.tfrecord'
tfrecord_test_name = 'movie_text2num_test.tfrecord'
ckpt_name = 'cbowgram_movie_embeddings.ckpt'
vocab_name = 'movie_vocab.pkl'

if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

model_checkpoint_path = os.path.join(data_folder_name, ckpt_name)
with open(os.path.join(data_folder_name, vocab_name), 'rb') as f:
    word_dict = pickle.load(f)
# load and generate data
stops = stopwords.words('english')
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
# load movie review data
texts, target = helper.load_movie_data()
# normalize data
texts = helper.normalize_text(texts, stops)
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]
texts = helper.text_to_numbers(texts, word_dict)
max_len = max([len(x) for x in texts])

train_indices = np.sort(np.random.choice(len(target), round(0.8 * len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = np.array([x for ix, x in enumerate(texts) if ix in train_indices])
texts_test = np.array([x for ix, x in enumerate(texts) if ix in test_indices])
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
# Split train/test set

x_train, x_test = texts_train, texts_test
y_train, y_test = target_train, target_test
vocab_size = len(word_dict.keys())
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# create an rnn model
x_input = tf.placeholder(shape=[None, max_sequence_length], dtype=tf.int32)
y_input = tf.placeholder(shape=[None], dtype=tf.int32)

embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
# embedding_mat = tf.constant(np.random.uniform(-1.0, 1.0, [vocab_size, embedding_size]).astype(np.float32))
x_embedding = tf.nn.embedding_lookup(embedding_mat, x_input)

with tf.variable_scope('weight') as scope:
    h_w = tf.get_variable(name="h_w", shape=[hidden_size, hidden_size], dtype=tf.float32)
    x_w = tf.get_variable(name="x_w", shape=[embedding_size, hidden_size], dtype=tf.float32)
    h_b = tf.get_variable(name="h_b", shape=[hidden_size], dtype=tf.float32)
    y_w = tf.get_variable(name="y_w", shape=[hidden_size, class_size], dtype=tf.float32)
    y_b = tf.get_variable(name="y_b", shape=[class_size], dtype=tf.float32)
    # self.x_b = tf.get_variable("x_b", tf.zeros(shape=[1, hidden_size], dtype=tf.float32))
    # scope.reuse_variables()


# shape of x_in = [batch size, embedding]
def rnn_cell(x_in, h_in=tf.zeros(shape=[1, hidden_size], dtype=tf.float32)):
    h_out = tf.add(tf.add(tf.matmul(h_in, h_w), h_b), tf.matmul(x_in, x_w))
    h_out = tf.tanh(h_out)
    return h_out


def rnn(x_embed=x_embedding):
    x_rnn = tf.split(axis=1, num_or_size_splits=max_sequence_length, value=x_embed)
    x_rnn = [tf.squeeze(x, [1]) for x in x_rnn]
    print(len(x_rnn))
    state = []
    for i in range(len(x_rnn)):
        if i == 0:
            state.append(rnn_cell(x_rnn[i]))
        else:
            state.append(rnn_cell(x_rnn[i], h_in=state[i-1]))
    last_o = state[-1]
    return state, last_o


_, last_output = rnn()

y_out = tf.add(tf.matmul(last_output, y_w), y_b)

# losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out, labels=y_input)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out, labels=y_input)
loss = tf.reduce_mean(losses)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_out, 1), tf.cast(y_input, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size)

    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]

        # Run train step
        train_dict = {x_input: x_train_batch, y_input: y_train_batch}
        sess.run(train_step, feed_dict=train_dict)

    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    # Run Eval Step
    test_dict = {x_input: x_test, y_input: y_test}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    print('Epoch: {}, Train Loss: {:.2}, Train Acc: {:.2}'.format(epoch+1, temp_train_loss, temp_train_acc))
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))