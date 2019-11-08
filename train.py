# encoding = 'utf-8'
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Dropout, LeakyReLU
from bert_serving.client import BertClient
from data_loader import Loader
import numpy as np
import re
import random

CORPUS_PATH = 'data\\證券交易法標記輸出檔'

bc = BertClient()

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
        if text[i] == '，' or text[i]== '。':
            idx += 1
        dic[i] = idx
    return dic

def generate_meaning_label(label, dic):
    tmp = {}
    for l in label:
        if dic[l['start_offset']] == dic[l['end_offset']]:
            tmp[dic[l['start_offset']]] = l['label']
        else:
            tmp[dic[l['start_offset']]] = l['label']
            tmp[dic[l['end_offset']]] = l['label']
    return tmp


def one_hot_label(ids, dic):
    ret = []
    for id in ids:
        res = [0 for _ in range(len(dic))]
        res[dic[id]] = 1
        ret.append(res)
    return ret


training_data = []
label_data = []
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
                training_data.append(s)
                label_data.append(l_dic[idx])
            else:
                training_data.append(s)
                label_data.append(0) # no label

# save label
label2id = {}
for id, l in enumerate(set(label_data)):
    label2id[l] = id
with open('label2id.txt', 'w', encoding='utf-8') as f:
    for k in label2id.keys():
        f.write('{} : {}\n'.format(label2id[k], k))

label_count = {}
for l in label_data:
    if str(l) in label_count:
        label_count[str(l)] += 1
    else:
        label_count[str(l)] = 1
print(label_count)
max_num = max([label_count[k] for k in label_count.keys()])


# using bert to generate article vector and use simple NN to predict the law category
print('start bert encoding')
t_X = bc.encode(training_data).tolist()

# balance data by copy data randomly for specific label until number is equal
# =====
tmp_X = []
tmp_Y = []

label2data= {}
tmp_label2data = {}


def random_generate(data, num):
    res = []
    for _ in range(num):
        res.append(data[random.randint(0,  len(data)-1)])
    return res


for i, x in enumerate(t_X):
    if label_data[i] in label2data:
        label2data[label_data[i]] += [x]
    else:
        label2data[label_data[i]] = [x]
for k in label2data.keys():
    tmp_X += random_generate(label2data[k], max_num)
    tmp_Y += [k for _ in range(max_num)]
# ====
tmp_Y = one_hot_label(tmp_Y, label2id)
train_X = np.asarray(tmp_X)
train_Y = np.asarray(tmp_Y)

print(train_X.shape)
print(train_Y.shape)

model = baseline_model(len(label2id))
model.summary()
cp = ModelCheckpoint('law2fact.h', monitor='val_loss', save_best_only=True)
model.fit(train_X, train_Y, epochs=20, batch_size=32, shuffle=True)
