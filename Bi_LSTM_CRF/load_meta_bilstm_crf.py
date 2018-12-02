import os
import tensorflow as tf
import pickle
import numpy as np
import helper
import jieba

data_folder_name = '..\\temp'
data_path_name = 'cn_nlp'
vocab_name = 'bilstm_crf_cn.pkl'
# ckpt_name = 'model_backup\\bilstm_crf_cn.ckpt'
ckpt_name = 'bilstm_crf_cn.ckpt'
ckpt_path = os.path.join(data_folder_name, data_path_name, ckpt_name)
vocab_path = os.path.join(data_folder_name, data_path_name, vocab_name)
test_str_list = ['这个仅仅是一个小测试',
                 '这仅仅是一个小测试',
                 '李小福是创新办主任也是云计算方面的专家',
                 '我相信自己一定能够实现它',
                 '今天你都干了些啥呀',
                 '实现祖国的完全统一是海内外全体中国人的共同心愿',
                 '南京市长江大桥',
                 '小时候买方便面先捏碎，再把调味包撒进去使劲捏然后吃',
                 '中文分词在中文信息处理中是最最基础的，无论机器翻译亦或信息检索还是其他相关应用，如果涉及中文，都离不开中文分词，因此中文分词具有极高的地位。',
                 '蔡英文和特朗普通话',
                 '研究生命的起源',
                 '他从马上下来',
                 '我马上下来',
                 '他站起身',
                 '老人家身体不错',
                 '老人家中很干净',
                 '他起身去北京',
                 '这的确定不下来',
                 '乒乓球拍卖完了',
                 '这家小商店的乒乓球拍卖完了',
                 '香港中文大学将来合肥一中进行招生宣传今年在皖招8人万家热线安徽第一门户',
                 '在伦敦奥运会上将可能有一位沙特阿拉伯的女子',
                 '美军中将竟公然说',
                 '在这些企业中国有企业有十个',
                 '锌合金把手的相关求购信息',
                 '北京大学生喝进口红酒',
                 '在北京大学生活区喝进口红酒',
                 '天真的你',
                 '我来到北京清华大学',
                 '我们中出了一个叛徒',
                 '将信息技术应用于教学实践']


test_str_len = []
for line in test_str_list:
    test_str_len.append(len(line))
max_len = max(test_str_len)


def viterbi_decode(score_, trans_):
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(score_, trans_)
    return viterbi_sequence


def evaluate(scores_, lengths_, trans_):
    pre_sequence = []
    for ix, score_ in enumerate(scores_):
        score_real = score_[:lengths_[ix]]
        pre_sequence.append(viterbi_decode(score_real, trans_))
    return pre_sequence


def split_cn(text_list_, pre_seq_, split_flag =' ', assist=False):
    strs_split = []
    if not assist:
        for ix, sentence in enumerate(text_list_):
            str_split = []
            for iw, word in enumerate(sentence):
                tag = pre_seq_[ix][iw]
                if tag == 0 or tag == 3:
                    str_split.append(word)
                    str_split.append(split_flag)
                else:
                    str_split.append(word)
            strs_split.append(''.join(str_split))
    else:
        for ix, sentence in enumerate(text_list_):
            str_split = []
            flag = False
            for iw, word in enumerate(sentence[::-1]):
                iw_r = len(sentence) - iw - 1
                tag = pre_seq_[ix][iw_r]
                if tag == 0:
                    str_split.append(split_flag)
                    str_split.append(word)
                    flag = False
                elif tag == 3:
                    if not flag:
                        str_split.append(split_flag)
                    str_split.append(word)
                    flag = True
                else:
                    str_split.append(word)
                    flag = False
            str_split.reverse()
            strs_split.append(''.join(str_split))
    return strs_split


# config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()

with open(vocab_path, 'rb') as f:
    word_dict = pickle.load(f)
vocabulary_size = len(word_dict)
word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))
saver = tf.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()
# name = [n.name for n in graph.as_graph_def().node]
# print(name)
x_input = graph.get_tensor_by_name('x_input:0')
seq_len = graph.get_tensor_by_name('seq_length:0')
dropout = graph.get_tensor_by_name('dropout:0')
trans_matrix = graph.get_tensor_by_name('crf_loss/transitions:0')
logits = graph.get_tensor_by_name('project/output/logits:0')
text2num = np.array(helper.text_to_numbers(test_str_list, word_dict, True))
text2num = np.array([(x+[0]*max_len)[0:max_len] for x in text2num])

feed_dict = {x_input: text2num, seq_len: test_str_len, dropout: 0}
pred_logits, trans = sess.run([logits, trans_matrix], feed_dict)
pred_sequence = evaluate(pred_logits, test_str_len, trans)
# print('trans', trans)
result = split_cn(test_str_list, pred_sequence, '/', True)
# result_r = split_cn(test_str_list, pred_sequence)
for line_a, line_b in zip(result, test_str_list):
    print(line_a)
# print(pred_sequence)
print('---------')
# for test_str in test_str_list:
#     seg_list = jieba.cut(test_str)
#     print('/'.join(seg_list))



