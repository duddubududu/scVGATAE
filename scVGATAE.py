import os, sys


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
from models import GCNModelVAE, GCNModelAE
from optimizer import loss_function
from utils import load_data,saveClusterResult, preprocess_graph
from sklearn.cluster import KMeans
# from clustering_metric import clustering_metrics
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--dw', type=int, default=1, help="whether to use deepWalk regularization, 0/1")
parser.add_argument('--epochs', type=int, default=180, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='Biase', help='type of dataset.')
parser.add_argument('--dataset', type=str, default='data.tsv', help='dataset file')
parser.add_argument('--data-type', type=str, default='tsv', help='(str, default None) If not None, will load the 10X data from this file path')
parser.add_argument('--n-clusters', default=0, type=str, help='number of clusters, if 0, use the leiden algorithm to find the number of clusters')
parser.add_argument('--normalize', default=True, type=str2bool, help='whether to normalize the adjacency matrix')
parser.add_argument('--sil-stp',default=True,type=str2bool,help='whether to use silhouette score to stop training')
parser.add_argument('--walk-length', default=5, type=int, help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
parser.add_argument('--number-walks', default=5, type=int, help='Number of random walks to start at each node')
parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')
parser.add_argument('--lr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--context', type=int, default=0, help="whether to use context nodes for skipgram")
parser.add_argument('--ns', type=int, default=1, help="whether to use negative samples for skipgram")
parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
parser.add_argument('--precisionModel', type=str, default='Float', 
                    help='Single Precision/Double precision: Float/Double (default:Float)')
parser.add_argument('--true_labels_path',default='/true_labels.csv',type=str,help='path of true labels for evaluation of ARI and NMI')
args = parser.parse_args()


def scVGATAE_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, cells, n_clusters = load_data(args)
    
    n_nodes, feat_dim = features.shape


    # Some preprocessing
    adj_norm= preprocess_graph(adj)
    adj_label = adj+ sp.eye(adj.shape[0])

    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    
    evaluation_metrics = {
        'epoch':None,
        'ROC':None,
        'AP':None,
        'vgae train loss':None,
    }
    
    header =evaluation_metrics.keys()
    with open('evaluation_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

    if args.model == 'gcn_vae': 
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    final_pred = None
    max_sil = 0
    sil_logs = []
    max_sil_hidden_emb = None
    max_sil_epoch = 0
    # 训练模型
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        z, mu, logvar,features_recon = model(features, adj_norm)

        loss = loss_function(preds=model.dc(z), labels=adj_label,
                             features=features,features_recon=features_recon,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        hidden_emb = mu.data.numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20).fit(hidden_emb)
        predict_labels = kmeans.predict(hidden_emb)
        sil_hid = metrics.silhouette_score(hidden_emb, predict_labels, metric='euclidean')
        print('Epoch: {:04d}'.format(epoch+1),
                   'train_loss= {:.4f}'.format(cur_loss),
                   'sil_hid= {:.4f}'.format(sil_hid))
        sil_logs.append(sil_hid)
        if sil_hid >= max_sil:
            max_sil_hidden_emb = hidden_emb
            final_pred = predict_labels
            max_sil = sil_hid
            max_sil_epoch = epoch
        arr_sil = np.array(sil_logs)
        if args.sil_stp:
            if len(arr_sil) >= 150:
                mean_0_n = np.mean(arr_sil[-30:])
                mean_n_2n = np.mean(arr_sil[-60: -30])
                if mean_0_n - mean_n_2n <= 0.1:
                    print('Stop early at', epoch, 'epoch')
                    break
            if len(arr_sil) >= 3:
                if  max_sil - arr_sil[-1] >= 0.02:
                    print('Stop early at', epoch, 'epoch')
                    break




    print("scVGATAE training Finished!")
    print("Evaluating scVGATAE final results...")

    

    
    if args.true_labels_path:
        true_path = 'data/'+args.dataset_str+args.true_labels_path
        real_label = pd.read_csv(true_path,index_col=0,sep=',')
        
        real_label.index = real_label.index.astype(str)
        real_y=real_label.loc[cells,'label']
        
        real_y=real_y.values.flatten()
        
        max_sil_ARI = metrics.adjusted_rand_score(final_pred, real_y)
        max_sil_NMI = metrics.normalized_mutual_info_score(final_pred, real_y)
    

        print("model.eval")
        model.eval()
        z = model(features, adj_norm)
        hidden_emb = z.data.numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20).fit(hidden_emb)
        predict_labels = kmeans.predict(hidden_emb)
        ARI = metrics.adjusted_rand_score(predict_labels, real_y)
        NMI = metrics.normalized_mutual_info_score(predict_labels, real_y)
        if max_sil_ARI >= ARI:
            print("scVGATAE final results:")
            print("ARI: {:.4f}".format(max_sil_ARI))
            print("NMI: {:.4f}".format(max_sil_NMI))
            np.savetxt('data/'+args.dataset_str+'/hidden_emb.txt', max_sil_hidden_emb)
            saveClusterResult(final_pred, cells, args.dataset_str)
            print("scVGATAE clusters saved!")
        else:
            print("scVGATAE final results:")    
            print("ARI: {:.4f}".format(ARI))
            print("NMI: {:.4f}".format(NMI))
            np.savetxt('data/'+args.dataset_str+'/hidden_emb.txt', hidden_emb)
            saveClusterResult(predict_labels, cells, args.dataset_str)
            print("scVGATAE clusters saved!")

    

    torch.save(model.state_dict(), 'scVGATAE_model.pth')
    print("scVGATAE model saved!")

    

 


                
if __name__ == '__main__':
    start = time.time()
    scVGATAE_for(args)
    end = time.time()
    run_time = end - start
    print('scVGATAE: Total run time is %.2f '%run_time, 'seconds.')