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

test = ['康燿則為萬國投信公司募集之「萬國鑽石證券投資信託基金」']

out = model.predict(bc.encode(test))
print(out)