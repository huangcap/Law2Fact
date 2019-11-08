###Fact 2 Law
use sentence-level classification to extract related label.

###flow
```buildoutcfg
sentence --bert-service--> sentence vector --NN model--> class label
```

###train
```buildoutcfg
input: json file folder
ex:
folder
|- a.json
|- b.json
|- c.json
...

output: model
```

###how to use
```
1.start bert-server(bert-serving-client==1.9.8 in your service system)
bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -num_worker=2
2.run train.py to get model
3.run test.py to use your model
```

###Ref
https://github.com/hanxiao/bert-as-service
##bert model
https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
