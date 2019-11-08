# encoding = 'utf-8'
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Dropout, LeakyReLU
from bert_serving.client import BertClient
from data_loader import Loader
import numpy as np
import re

bc = BertClient()
model = load_model('law2fact.h')

test = ['我想吃蘋果']

out = model.predict(bc.encode(test))
print(out)