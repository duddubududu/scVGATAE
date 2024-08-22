import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
dataset='Adipose'
data_path = 'data/'+dataset+'/hidden_emb.txt'
pred_path = 'data/'+dataset+'/pred_labels.csv'
umap_path = 'data/'+dataset+'/umap_embedding.txt'
hidden = pd.read_csv(data_path, sep=' ', index_col=None, header=None)
print(hidden.shape)
pred_path= pd.read_csv(pred_path,index_col=0,sep=',')
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
embedding = reducer.fit_transform(hidden.values)
df_embedded = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
df_embedded['Label'] = pd.Series(pred_path.values.flatten(), index=None).astype('category')
df_embedded.to_csv(umap_path, sep='\t')