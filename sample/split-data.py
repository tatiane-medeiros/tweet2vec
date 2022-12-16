import pandas as pd
import numpy as np
import re

stop_words_pt = ['a', 'ao', 'aos', 'as', 'às', 'com', 'como', 'da', 'das', 'de', 'do', 'dos', 'em',
                 'mais', 'na', 'não', 'nas', 'nem', 'no', 'nos', 'num', 'numa', 'o', 'os', 'ou',
                 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'que', 'sem', 'um', 'uma']

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z0-9/,.]+', ' ', text)
    # remove pontuação
    text = re.sub(r'(?<=[^0-9])([^a-z0-9])', r' ', text)
    text = re.sub(r'/(?=[^0-9])', r' ', text)
    text = re.sub('[.,]', '',text)

    # separa numeros e letras
    text = re.sub(r'(?<=[0-9])(?=[a-z])', r' ', text)
    text = re.sub(r'(?<=[a-z])(?=[0-9])', r' ', text)
    # substitui numeros
    text = re.sub('\d+', '9', text)
    text = re.sub('\d/\d/\d', '', text)
    text = re.sub(' +', ' ', text).strip()
    return text

df = pd.read_csv('train.csv')

# pre processamento
df.iloc[:, 1] = df.iloc[:, 1].apply(clean_text)
#np.savetxt('data.txt', df.values, fmt='%s', delimiter='\t', encoding='utf-8')
# np.savetxt('encoder.txt', df.iloc[:, 1].values, fmt='%s', encoding='utf-8')

#df = df.drop(df[df['Unidade'] == 'indefinido'].index)

test = df.sample(frac=0.4)
train = df.drop(test.index)
# %%

np.savetxt('../data/train.txt', train.values, fmt='%s', delimiter='\t', encoding='utf-8')
np.savetxt('../data/test.txt', test.values, fmt='%s', delimiter='\t', encoding='utf-8')
np.savetxt('../data/encoding.txt', test.iloc[:, 1].values, fmt='%s', encoding='utf-8')
