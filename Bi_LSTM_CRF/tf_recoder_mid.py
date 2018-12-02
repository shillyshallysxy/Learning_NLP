import helper
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from nltk.corpus import stopwords
import pickle
import os
ops.reset_default_graph()

sess = tf.Session()
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp'
vocab_name = 'bilstm_crf_cn.pkl'
train_record_name = 'people_daily_train.tfrecord'
test_record_name = 'people_daily_test.tfrecord'
data_name = '199801out.txt'
data_name2 = '2014out.txt'
vocab_path = os.path.join(data_folder_name, data_path_name, vocab_name)

with open(vocab_path, 'rb') as f:
    word_dict = pickle.load(f)
vocabulary_size = len(word_dict)


# load movie review data
texts, targets, lengths = helper.load_data(data_name, data_name2)
max_len = max(lengths)
print(max_len)
texts = helper.text_to_numbers(texts, word_dict, True)
train_indices = np.random.choice(len(targets), round(0.9 * len(targets)), replace=False)
test_indices = np.sort(np.array(list(set(range(len(targets))) - set(train_indices))))
texts_train = np.array(texts)[train_indices]
texts_test = np.array(texts)[test_indices]
target_train = np.array(targets)[train_indices]
target_test = np.array(targets)[test_indices]
lengths_train = lengths[train_indices]
lengths_test = lengths[test_indices]


def write_binary(record_name, texts_, target_, lens_):
    record_path = os.path.join(data_folder_name, data_path_name, record_name)
    writer = tf.python_io.TFRecordWriter(record_path)
    for it, text in enumerate(texts_):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=target_[it])),
                    "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[lens_[it]]))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "label": tf.VarLenFeature(tf.int64),
                                                              "length": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    text_ = tf.sparse_tensor_to_dense(features["text"])
    label_ = tf.sparse_tensor_to_dense(features["label"])
    lens_ = tf.cast(features["length"], tf.int32)
    return text_, label_, lens_


write_binary(train_record_name, texts_train, target_train, lengths_train)
write_binary(test_record_name, texts_test, target_test, lengths_test)

