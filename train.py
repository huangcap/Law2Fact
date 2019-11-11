# encoding = 'utf-8'
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Dropout, LeakyReLU
from bert_serving.client import BertClient
from data_loader import Loader
import numpy as np
import re
import random
import json

CORPUS_PATH = 'data\\證券交易法標記輸出檔_v1'

bc = BertClient()
print('bert is ready')
dl = Loader()
dl.load_corpus(CORPUS_PATH)


def baseline_model(category_out):
    # create model
    model = Sequential()
    model.add(Dense(768, input_dim=768, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, input_dim=768, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(category_out, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# prepare training data
def generate_sentence2pos_dict(text):
    dic = {}
    idx = 0
    for i in range(len(text)):
        if text[i] == '，' or text[i] == '。':
            idx += 1
        dic[i] = idx
    return dic


def generate_meaning_label(label, dic):
    tmp = {}
    for l in label:
        if dic[l['start_offset']] == dic[l['end_offset']-1]:
            tmp[dic[l['start_offset']]] = l['label']
        else:
            for i in range(dic[l['start_offset']], dic[l['end_offset']-1]+1):
                tmp[i] = l['label']
    return tmp


def one_hot_label(ids, dic):
    ret = []
    for id in ids:
        res = [0 for _ in range(len(dic))]
        res[dic[id]] = 1
        ret.append(res)
    return ret


training_data = []
# label_data = []
# dl.text = [dl.text[0]]
for i in range(len(dl.text)):
    text = dl.text[i]
    label = dl.label[i]
    sentences = re.split('[，。]', text)
    dic = generate_sentence2pos_dict(text)
    l_dic = generate_meaning_label(label, dic)
    for idx, s in enumerate(sentences):
        if(s):
            if idx in l_dic:
                training_data.append([s, l_dic[idx]])
                # label_data.append(l_dic[idx])
            else:
                training_data.append([s, 0])
                # label_data.append(0) # no label

label2id = {}
id = 0
for l in (training_data):
    if l[1] not in label2id:
        label2id[l[1]] = id
        id += 1
    else:
        pass

with open('label2id.txt', 'w', encoding='utf-8') as f:
    for k in label2id.keys():
        f.write('{} : {}\n'.format(label2id[k], k))

# encode first
tmp_data = bc.encode([data[0] for data in training_data])
for data, encode in zip(training_data, tmp_data):
    # print(bc.encode([data[0]]))
    data[0] = encode

# write training_data
with open('training_data', 'w', encoding='utf-8') as f:
    j = {}
    j['data'] = []
    for d in training_data:
        tmp = {}
        tmp['code'] = d[0].tolist()
        tmp['label'] = d[1]
        j['data'].append(tmp)
    json.dump(j, f)

# show stat
stat_dict = {}
for d in training_data:
    if d[1] in stat_dict:
        stat_dict[d[1]] += 1
    else:
        stat_dict[d[1]] = 1
for k in stat_dict:
    print('{} {}'.format(k, stat_dict[k]))

# save label
# label2id = {}
# for id, l in enumerate(set(training_data)):
#     label2id[l[1]] = id
# with open('label2id.txt', 'w', encoding='utf-8') as f:
#     for k in label2id.keys():
#         f.write('{} : {}\n'.format(label2id[k], k))
#
# label_count = {}
# for l in training_data:
#     if l[1] in label_count:
#         label_count[l[1]] += 1
#     else:
#         label_count[l[1]] = 1
# print(label_count)
# max_num = max([label_count[k] for k in label_count.keys()])
#
#
# using bert to generate article vector and use simple NN to predict the law category
# print('start bert encoding')
# t_X = bc.encode(training_data).tolist()
#
# balance data by copy data randomly for specific label until number is equal
# =====
# tmp_X = []
# tmp_Y = []
#
# label2data= {}
# tmp_label2data = {}
#
#
# def random_generate(data, num):
#     res = []
#     for _ in range(num):
#         res.append(data[random.randint(0,  len(data)-1)])
#     return res
#
#
# for i, x in enumerate(t_X):
#     if training_data[i][1] in label2data:
#         label2data[training_data[i][1]] += [x]
#     else:
#         label2data[training_data[i][1]] = [x]
# for k in label2data.keys():
#     tmp_X += random_generate(label2data[k], max_num)
#     tmp_Y += [k for _ in range(max_num)]
# tmp_Y = one_hot_label(tmp_Y, label2id)
# train_X = np.asarray(tmp_X)
# train_Y = np.asarray(tmp_Y)
# print(train_X.shape)
# print(train_Y.shape)


# ====
# class DataGenerator(object):
#     def __init__(self, rescale=None):
#         self.train = []
#         self.target = []
#         self.train_sentences = []
#         self.reset()
#
#     def reset(self):
#         self.train = []
#         self.target = []
#         self.train_sentences = []
#
#     def flow_from_directory(self, data, label2id, batch_size=32):
#         input_data = np.zeros(
#             (batch_size, 768),
#             dtype='float32')
#         target_data = np.zeros(
#             (batch_size, len(label2id)),
#             dtype='float32')
#         while True:
#             time = 0
#             for i, d in enumerate(data):
#                 input_data[time] = d[0]
#                 # self.train_sentences.append(d[0])
#                 target_data[time, label2id[d[1]]] = 1.
#                 time += 1
#                 # print(input_data)
#                 # print(target_data)
#                 if time == batch_size:
#                     self.train = input_data
#                     self.target = target_data
#                     inputs = np.asarray(self.train, dtype='float32')
#                     targets = np.asarray(self.target, dtype='float32')
#                     self.reset()
#                     time = 0
#                     # print('in: '+str(inputs[0]))
#                     # print('out: '+str(targets[0]))
#                     yield inputs, targets
#             self.train = input_data
#             self.target = target_data
#             inputs = np.asarray(self.train, dtype='float32')
#             targets = np.asarray(self.target, dtype='float32')
#             self.reset()
#             yield inputs, targets


# EPOCHS = 20
# BATCH = 128
# STEPS_PER_EPOCH = (len(training_data)/BATCH)
# datagen = DataGenerator()
# model = baseline_model(len(label2id))
# model.summary()
# # cp = ModelCheckpoint('law2fact.h', monitor='val_loss', save_best_only=True)
# # model.fit(train_X, train_Y, epochs=20, batch_size=32, shuffle=True)
# model.fit_generator(
#     generator=datagen.flow_from_directory(training_data, label2id, BATCH),
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     )
#
# model.save('law2fact.h')