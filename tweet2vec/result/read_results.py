import numpy as np
import _pickle as pkl
import pandas as pd

##f = open('data.pkl', 'rb')
##data = pkl.load(f)
##f.close()
##f = open('targets.pkl', 'rb')
##targets = pkl.load(f)
##f.close()
##
##for i in range(len(targets)):
##    print(targets[i], '-', data[i])


f = open('readable.txt', 'r')
a = f.read()
f.close()
b = [x.split('\t') for x in a.split('\n')]

validation = pd.DataFrame(b, columns=['un. medida','predicted','texto'])
