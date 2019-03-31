
import os
path=os.path.abspath(os.path.dirname(__file__))
os.chdir(path)

import GCNs
import exps
import results
import log
from data_process import *

import os
path=os.path.abspath(os.path.dirname(__file__))
os.chdir(path)

model_input={}
model_input['settings']={'dataset':'pubmed','training_epoch':200,'early_stopping':100,
                       'hidden1':16,'learning_rate':0.01 ,'weight_decay':5e-1 ,
                       'dropout':0.5,
                       'all_test':False}
model_input.update(get_dataset(model_input['settings']['dataset']))
#GCNs.sb
#print(dir(GCNs))
exp1_model=GCNs.GCN_kipf(model_input)
description='A little test for new framework on GCN'
recorder=statistic_recorder()

for i in range(1):
    result=exp1_model.run()
    recorder.insert(result)


write_result(description,exp1_model.settings,recorder.statistic)













