import numpy as np
import _pickle as pkl
import pandas as pd

def targets():
    f = open('data.pkl', 'rb')
    data = pkl.load(f)
    f.close()
    f = open('targets.pkl', 'rb')
    targets = pkl.load(f)
    f.close()
    table = []
    for i in range(len(targets)):
       table.append([targets[i], data[i]])

    return table

def test_table():

    f = open('readable.txt', 'r')
    a = f.read()
    f.close()
    b = [x.split('\t') for x in a.split('\n')]

    validation = pd.DataFrame(b, columns=['un. medida','predicted','texto'])
    validation.sort_values('un. medida').to_csv('result.csv', index=False)

def predicted_table():
    f = open('predicted_tags.txt', 'r')
    a = f.read()
    f.close()
    a = a.split('\n')

    f = open('../../data/data.txt', 'r')
    b = f.read()
    f.close()
    c = [x.split('\t') for x in b.split('\n')]
    data = []

    for i in range(len(c)):
        data.append(a[i].split()[:1] + c[i])

    return data

d = predicted_table()
predict = pd.DataFrame(d, columns=['predicted','unid. real','texto'])
#predict.sort_values('unid. real').to_csv('predicted.csv', index=False)
