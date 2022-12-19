# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:22:03 2022

@author: Tatiane
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
import seaborn as sns


result_df = pd.read_csv('predicted.csv')
result_df = result_df.dropna()
y = result_df['predicted']
x = result_df['unid. real']

# %%
np.errstate(divide='ignore',invalid='ignore')
np.seterr(divide='ignore', invalid='ignore')

acc = accuracy_score(x,y)
print('accuracy: ', acc)

pr = precision_score(x,y, average=None)
print('precision: ', pr)

recall = recall_score(x,y, average=None)
print('recall: ', recall.mean())



# %%

labels = ['indefinido', 'ml','comp', 'grama', 'unidade','litro','kg','metro','m3', 'caixa','ampola','frasco','conjunto','kit', 'cilindro', 'rolo','galao']
mapping = dict(zip(labels, range(0,17)))

x = x.map(mapping).fillna(0)
y = y.map(mapping).fillna(0)
#labels = [labels[int(i)] for i in list(np.unique(y))]

report = classification_report(x,y)
f = open("data_report.txt",'w')
f.write(report)
f.close()

# %%
mc=confusion_matrix(x,y)

np.errstate(divide='ignore',invalid='ignore')
np.seterr(divide='ignore', invalid='ignore')

mcn = mc.astype('float') / mc.sum(axis=1)[:, np.newaxis]
mcn = np.nan_to_num(mcn, copy=True, nan=0, posinf=None, neginf=None)

sns.set_theme(palette="viridis")
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(mcn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)


plt.ylabel('Valores reais')
plt.xlabel('Classificados pelo tweet2vec')
plt.yticks(rotation=0)
plt.xticks(rotation=-30)
#plt.show(block=False)
plt.savefig('t2v-confusion-matrix.pdf')