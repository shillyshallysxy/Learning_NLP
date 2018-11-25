import tensorflow as tf
import pickle
import os
import numpy as np
import helper
sess = tf.Session()
max_len = 39
data_folder_name = 'temp'
vocab_name = 'movie_vocab.pkl'
tfrecord_test_name = 'movie_text2num_test1code.tfrecord'

with open(os.path.join(data_folder_name, vocab_name), 'rb') as f:
    word_dict = pickle.load(f)
ckpt_name = 'movie_emo_classifier_cnn_adam.ckpt'
ckpt_path = os.path.join(data_folder_name, ckpt_name)
saver = tf.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()
handle = graph.get_tensor_by_name('handle:0')
predict = graph.get_tensor_by_name('predict:0')


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


test_str = ['i like watch this movies']
test_str = helper.text_to_numbers(test_str, word_dict)
write_binary(tfrecord_test_name, test_str, [1])
data_set_test = get_dataset(tfrecord_test_name)
data_set_test = data_set_test.padded_batch(1, ([max_len], []))
data_set_test_iter = data_set_test.make_one_shot_iterator()
handle_test = sess.run(data_set_test_iter.string_handle())
feed_dict = {handle: handle_test}

print(sess.run(predict, feed_dict))

os.remove(os.path.join(data_folder_name, tfrecord_test_name))
# name = [n.name for n in graph.as_graph_def().node]
# print(name)


