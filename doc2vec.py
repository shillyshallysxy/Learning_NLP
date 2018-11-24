import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import helper
from nltk.corpus import stopwords
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Make a saving directory if it doesn't exist
data_folder_name = 'temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

# Declare model parameters
batch_size = 500
vocabulary_size = 7500
generations = 100000
model_learning_rate = 0.001

embedding_size = 200   # Word embedding size
doc_embedding_size = 100   # Document embedding size
concatenated_size = embedding_size + doc_embedding_size

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 3       # How many words to consider to the left.

# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 500

# Declare stop words
stops = stopwords.words('english')
# stops = []

# We pick a few test words for validation.
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
# Later we will have to transform these into indices

# pre process of data
print('Creating Dictionary')
texts, target = helper.load_movie_data()
texts = helper.normalize_text(texts, stops)
texts = [x for x in texts if len(x.split()) > window_size]
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
word_dict = helper.build_dictionary(texts, vocabulary_size)
word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))
text_data = helper.text_to_numbers(texts, word_dict)
valid_data = [word_dict[x] for x in valid_words]

# build model
print('Creating Model')
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1., 1.))
doc_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, doc_embedding_size], -1., 1.))
nce_weight = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                             stddev=1.0 / np.sqrt(concatenated_size)))
nce_bias = tf.Variable(tf.zeros([vocabulary_size]))
x_input = tf.placeholder(shape=[batch_size, window_size+1], dtype=tf.int32)
y_input = tf.placeholder(shape=[batch_size, 1], dtype=tf.int32)
valid_input = tf.constant(valid_data, dtype=tf.int32)
doc_indices = tf.slice(x_input, [0, window_size], [batch_size, 1])
embed = tf.zeros([batch_size, embedding_size])
for i in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_input[:, i])
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)
final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(doc_embed)])
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                     biases=nce_bias,
                                     labels=y_input,
                                     inputs=final_embed,
                                     num_classes=vocabulary_size,
                                     num_sampled=num_sampled))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})
# validate test
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
norm_embeddings = embeddings/norm
valid_embed = tf.nn.embedding_lookup(norm_embeddings, valid_input)
similarity = tf.matmul(valid_embed, norm_embeddings, transpose_b=True)
# train model
print('Starting Training')
for i in range(generations):
    x_data, y_data = helper.generate_batch_data(text_data, batch_size,
                                                window_size,  method='doc2vec')
    feed_dict = {x_input: x_data, y_input: y_data}
    sess.run(train_step, feed_dict=feed_dict)
    if (i+1) % print_loss_every == 0:
        print('Loss at step {} : {}'.format(i + 1, sess.run(loss, feed_dict=feed_dict)))
    if (i+1) % print_valid_every == 0:
        top_k = 5
        sim = sess.run(similarity, feed_dict=feed_dict)
        for k in range(len(valid_words)):
            sim_indices = (-sim[k, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(word_dict_rev[valid_data[k]])
            for j in range(top_k):
                log_str = "{} {},".format(log_str, word_dict_rev[sim_indices[j]])
            print(log_str)
    if (i+1) % save_embeddings_every == 0:
        save_path = os.path.join(data_folder_name, 'movie_vocab.pkl')
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'doc2vec_movie_embeddings.ckpt')
        saver.save(sess, model_checkpoint_path)
        with open(save_path, 'wb') as f:
            pickle.dump(word_dict, f)
        print('Model saved in file: {}'.format(save_path))