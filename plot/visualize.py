# Visualize using Matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
plt.figure(figsize=(10, 8))
# read the data
dataset='Adipose'
umap_path = 'data/'+dataset+'/umap_embedding.txt'
umap_embedded= pd.read_csv(umap_path , index_col=0, sep='\t')
umap_embedded['Label']=umap_embedded['Label'].astype('category')
sns.scatterplot(data=umap_embedded, x='UMAP1', y='UMAP2', hue='Label')
'''
Change the position of the legend to the upper left corner
'''

ax = plt.gca()

ax.legend(loc='lower left')
'''
Change the label of the legend
'''

handles, labels = ax.get_legend_handles_labels()
 
new_labels = [label.replace('0', 'Adipocyte_SPP1 high').replace('1', 'Mast cell').replace('2', 'NewCellType2').replace('3', 'Stromal cell').replace('4', 'Adipocyte _FGR high').replace('5', 'Neutrophil').replace('6', 'Proliferating cell').replace('7', 'Unknown') for label in labels]
 
ax.legend(handles, new_labels, loc='lower left')
plt.title('scVGATAE')
plt.xlabel('')
plt.ylabel('Adult-Adipose cells')
plt.show()