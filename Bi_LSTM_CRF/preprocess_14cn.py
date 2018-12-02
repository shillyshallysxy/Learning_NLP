# -*-coding:utf-8-*-
# Chinese Characters: B(Begin),E(End),M(Middle),S(Single)
import codecs
import os


def character_tagging(input_file_, output_file_):
    input_data = codecs.open(input_file_, 'r', 'utf-8')
    output_data = codecs.open(output_file_, 'w', 'utf-8')
    for line in input_data.readlines():
        # 移除字符串的头和尾的空格。strip()方法默认是移除空格的
        word_list = line.strip().split()
        for word in word_list:
            # words = word.split("/")
            word = word
            if len(word) == 1:
                if word == '。' or word == '？' or word == '！':
                    output_data.write("\n")
                else:
                    output_data.write(word + "/S ")
            elif len(word) >= 2:
                output_data.write(word[0] + "/B ")
                for w in word[1: len(word) - 1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word) - 1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()


data_root_name = '..\\temp'
data_folder_name = 'cn_nlp'
data_name = 'news_benchmark.txt'
data_output_name = 'news_benchmark_out.txt'
input_file = os.path.join(data_root_name, data_folder_name, data_name)
output_file = os.path.join(data_root_name, data_folder_name, data_output_name)
character_tagging(input_file, output_file)
