import pandas as pd

train_df = pd.read_csv('und_amostra.csv')

file = open('classes.txt', 'r')
data = file.read().split('\n')
file.close()
data = [x.split() for x in data][1:]
cat = {x[0]: x[1] for x in data}

new_df = train_df[[train_df.columns[2],train_df.columns[1]]]
new_df['Unidade'] = new_df['Unidade'].map(cat).fillna("indefinido")

# %% save
new_df.to_csv('train.csv', index=False)
