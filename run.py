#!/usr/bin/env python

from model import AITAClassifier, train, evaluate
from data.read_data import Data 

data = Data.from_pkl()
data.one_hot()
model = AITAClassifier(data)
train(model,data,batch_size=16)
evaluate(model,data)
