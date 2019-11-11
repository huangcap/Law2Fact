# encoding = 'utf-8'
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Dropout, LeakyReLU
import numpy as np
import re
import random
import json

EPOCHS = 20
BATCH = 128


class DataGenerator(object):
    def __init__(self, rescale=None):
        self.train = []
        self.target = []
        self.train_sentences = []
        self.reset()

    def reset(self):
        self.train = []
        self.target = []
        self.train_sentences = []

    def flow_from_directory(self, data, label2id, batch_size=32):
        input_data = np.zeros(
            (batch_size, 768),
            dtype='float32')
        target_data = np.zeros(
            (batch_size, len(label2id)),
            dtype='float32')
        while True:
            time = 0
            for i, d in enumerate(data):
                # print(d['code'])
                # print(d['label'])
                input_data[time] = np.asarray(d['code'])
                target_data[time, label2id[d['label']]] = 1.
                time += 1
                if time == batch_size:
                    self.train = input_data
                    self.target = target_data
                    inputs = np.asarray(self.train, dtype='float32')
                    targets = np.asarray(self.target, dtype='float32')
                    self.reset()
                    time = 0
                    # print('in: '+str(inputs[0]))
                    # print('out: '+str(targets[0]))
                    yield inputs, targets
            self.train = input_data
            self.target = target_data
            inputs = np.asarray(self.train, dtype='float32')
            targets = np.asarray(self.target, dtype='float32')
            self.reset()
            yield inputs, targets


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


with open('training_data', 'r', encoding='utf-8') as f:
    j = json.loads(f.read())

label2id = {}
with open('label2id.txt', 'r', encoding='utf-8') as f:
    tmp = f.read().split('\n')
    for l in tmp:
        if l:
            label = l.split(':')[1].strip()
            id = l.split(':')[0].strip()
            label2id[int(label)] = int(id)


STEPS_PER_EPOCH = len(j['data'])/BATCH

datagen = DataGenerator()
model = baseline_model(len(label2id))
model.summary()
# cp = ModelCheckpoint('law2fact.h', monitor='val_loss', save_best_only=True)
# model.fit(train_X, train_Y, epochs=20, batch_size=32, shuffle=True)
model.fit_generator(
    generator=datagen.flow_from_directory(j['data'], label2id, BATCH),
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    )

model.save('law2fact.h')