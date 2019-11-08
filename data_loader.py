import glob
import json


class Loader:
    def __init__(self):
        self.text = []
        self.label = []
        self.article = ''

    def load_corpus(self, idx): # for training
        f = glob.glob(idx+'\\*.json')
        for fn in f:
            file = open(fn, 'r', encoding='utf-8')
            j = json.loads(file.read())
            self.text.append(j['text'])
            self.label.append(j['annotations'])
            file.close()

