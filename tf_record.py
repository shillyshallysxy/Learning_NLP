import helper
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from nltk.corpus import stopwords
import pickle
import os
ops.reset_default_graph()
data_folder_name = 'temp'
# tfrecord_name = 'movie_text2num.tfrecord'
tfrecord_train_name = 'movie_text2num_train.tfrecord'
tfrecord_test_name = 'movie_text2num_test.tfrecord'
ckpt_name = 'cbowgram_movie_embeddings.ckpt'
save_ckpt_name = 'movie_emo_classifier_cnn_adam.ckpt'
vocab_name = 'movie_vocab.pkl'

if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

model_checkpoint_path = os.path.join(data_folder_name, ckpt_name)
with open(os.path.join(data_folder_name, vocab_name), 'rb') as f:
    word_dict = pickle.load(f)

sess = tf.Session()
batch_size = 500
test_batch_size = 2000
embedding_size = 200
vocabulary_size = 253703
n_gram = 2
num_channels = 1
conv_features = 64
sentence_size = 32
generations = 2000
learning_rate = 0.0005
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
embeddings = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=0.1))
saver = tf.train.Saver({"embeddings": embeddings})
saver.restore(sess, model_checkpoint_path)


def write_binary(tf_record_name, texts_, target_):
    writer = tf.python_io.TFRecordWriter(os.path.join(data_folder_name, tf_record_name))
    for it, text in enumerate(texts_):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_[it]]))}
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "label": tf.FixedLenFeature([], tf.int64)})
    text = tf.sparse_tensor_to_dense(features["text"])
    label = tf.cast(features["label"], tf.int32)
    return text, label


def get_dataset(tf_record_name):
    dataset = tf.data.TFRecordDataset(os.path.join(data_folder_name, tf_record_name))
    return dataset.map(__parse_function)


# train_indices = np.sort(np.random.choice(len(target), round(0.8 * len(target)), replace=False))
# test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
# texts_train = np.array([x for ix, x in enumerate(texts) if ix in train_indices])
# texts_test = np.array([x for ix, x in enumerate(texts) if ix in test_indices])
# target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
# target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
# write_binary(tfrecord_name, texts, target)
# write_binary('movie_text2num_train.tfrecord', texts_train, target_train)
# write_binary('movie_text2num_test.tfrecord', texts_test, target_test)
# exit()
train_data_set = get_dataset(tfrecord_train_name)
train_data = train_data_set.shuffle(5000).padded_batch(batch_size, padded_shapes=([max_len], [])).repeat()
train_iter = train_data.make_one_shot_iterator()
train_handle = sess.run(train_iter.string_handle())

test_data_set = get_dataset(tfrecord_test_name)
test_data = test_data_set.padded_batch(test_batch_size, padded_shapes=([max_len], [])).repeat()
test_iter = test_data.make_one_shot_iterator()
test_handle = sess.run(test_iter.string_handle())

# create placeholder
handle = tf.placeholder(tf.string, shape=[], name='handle')
iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
x_, y_ = iterator.get_next()
x_embed = tf.nn.embedding_lookup(embeddings, x_)
print(tf.shape(x_embed))
y_ = tf.cast(tf.expand_dims(y_, 1), tf.float32)


# build model
def build_model_cnn(x_in):
    x_in = tf.expand_dims(x_in, 3)
    conv1_weight = tf.Variable(tf.truncated_normal(shape=[n_gram, embedding_size, num_channels, conv_features],
                                                   stddev=0.05, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros(shape=[conv_features], dtype=tf.float32))

    full1_weight = tf.Variable(tf.truncated_normal(shape=[conv_features, sentence_size], dtype=tf.float32))
    full1_bais = tf.Variable(tf.truncated_normal(shape=[1, sentence_size], dtype=tf.float32))

    W_weight = tf.Variable(tf.truncated_normal(shape=[sentence_size, 1], dtype=tf.float32))
    b_bais = tf.Variable(tf.truncated_normal(shape=[1, 1], dtype=tf.float32))

    conv1 = tf.nn.conv2d(x_in, conv1_weight, strides=[1, 1, 1, 1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    # max_pool = tf.reduce_max(relu1, axis=1, keepdims=True)
    max_pool = tf.nn.max_pool(relu1, [1, max_len+1-n_gram, 1, 1], [1, 1, 1, 1], padding="VALID")
    full1_input = tf.reshape(max_pool, [-1, conv_features])
    sentence_vec_output = tf.nn.relu(tf.add(tf.matmul(full1_input, full1_weight), full1_bais))
    model_output_ = tf.add(tf.matmul(sentence_vec_output, W_weight), b_bais)
    return model_output_


def build_model_sentence2vec_avg(x_in):
    embed_avg = tf.reduce_mean(x_in, 1)
    A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    # Declare logistic model (sigmoid in loss function)
    model_output_ = tf.add(tf.matmul(embed_avg, A), b)
    return model_output_


class Rnn_Model():
    def __init__(self, x_in):
        self.hidden_size = 64
        self.class_size = 1
        self.x_in = x_in
        with tf.variable_scope('weight') as scope:
            h_w = tf.get_variable(name="h_w", shape=[self.hidden_size, self.hidden_size], dtype=tf.float32)
            x_w = tf.get_variable(name="x_w", shape=[embedding_size, self.hidden_size], dtype=tf.float32)
            h_b = tf.get_variable(name="h_b", shape=[self.hidden_size], dtype=tf.float32)
            y_w = tf.get_variable(name="y_w", shape=[self.hidden_size, self.class_size], dtype=tf.float32)
            y_b = tf.get_variable(name="y_b", shape=[self.class_size], dtype=tf.float32)

        # shape of x_in = [batch size, embedding]
        def rnn_cell(x_in_, h_in=tf.zeros(shape=[1, self.hidden_size], dtype=tf.float32)):
            h_out = tf.add(tf.add(tf.matmul(h_in, h_w), h_b), tf.matmul(x_in_, x_w))
            h_out = tf.tanh(h_out)
            return h_out

        def rnn(x_in_):
            x_rnn = tf.split(axis=1, num_or_size_splits=max_len, value=x_in_)
            x_rnn = [tf.squeeze(x, [1]) for x in x_rnn]
            state = []
            for i in range(len(x_rnn)):
                if i == 0:
                    state.append(rnn_cell(x_rnn[i]))
                else:
                    state.append(rnn_cell(x_rnn[i], h_in=state[i - 1]))
            last_o = state[-1]
            return state, last_o

        _, last_output = rnn(self.x_in)
        self.model_output = tf.add(tf.matmul(last_output, y_w), y_b)


class LSTM_Model():
    def __init__(self, x_in_):
        self.embedding_size = embedding_size
        self.rnn_size = 32
        self.batch_size = batch_size
        self.training_seq_len = max_len
        self.vocab_size = 1
        # self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.lstm_cell = tf.contrib.rnn.LSTMCell(name='basic_lstm_cell', num_units=self.rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        output, final_state = tf.nn.dynamic_rnn(self.lstm_cell, x_in_, dtype=tf.float32)
        output = tf.nn.dropout(output, 0.5)
        final_layer = output[:, -1, :]
        prev_transformed = tf.matmul(final_layer, W) + b
        # Get the index of the output (also don't run the gradient)
        # self.model_output = tf.stop_gradient(prev_transformed)
        self.model_output = prev_transformed


# model_output = build_model_cnn(x_embed)
model_output = LSTM_Model(x_embed).model_output

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, tf.trainable_variables()), 4.5)
# train_step = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

prediction = tf.round(tf.sigmoid(model_output), name='predict')
predictions_correct = tf.cast(tf.equal(prediction, y_), tf.float32)
accuracy = tf.reduce_mean(predictions_correct, name='accuracy')
init = tf.global_variables_initializer()
sess.run(init)
print('Starting Logistic Doc2Vec Model Training')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
max_acc = 0
saver_meta = tf.train.Saver(max_to_keep=1)
for i in range(generations):
    feed_dict = {handle: train_handle}
    sess.run(train_step, feed_dict=feed_dict)

    if (i + 1) % 50 == 0:
        i_data.append(i + 1)
        test_feed_dict = {handle: test_handle}
        train_loss_temp = sess.run(loss, feed_dict=feed_dict)
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict=test_feed_dict)
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict=feed_dict)
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict=test_feed_dict)
        test_acc.append(test_acc_temp)
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): '
              '{:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
        # print(sess.run([tf.squeeze(prediction), tf.squeeze(y_)], feed_dict=test_feed_dict))
        if (i+1) >= 1000 and max_acc <= test_acc_temp:
            max_acc = test_acc_temp
            saver_meta.save(sess, os.path.join(data_folder_name, save_ckpt_name))
            print('model saved')

