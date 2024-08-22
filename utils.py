import pickle as pkl
import anndata as ad
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import scanpy as sc
import pandas as pd
import h5py
def load_data(args):
    data_path = 'data/'+args.dataset_str+'/'+args.dataset
    if args.data_type == 'mtx':
        adata = sc.read_10x_mtx(data_path, var_names='gene_symbols', cache=True)
    elif args.data_type == 'tsv':

        data = pd.read_csv(data_path, sep='\t', index_col=0)
        data = data.T  
        adata = sc.AnnData(X=data.values)
        adata.obs_names = data.index.tolist()  
        adata.var_names = data.columns.tolist()  
    elif args.data_type == 'h5':
        data_mat = h5py.File(data_path, 'r')
        x = np.array(data_mat['X']).astype('float64')
        data_mat.close()
        adata = sc.AnnData(x, dtype="float64")
    elif args.data_type == 'txt':
        data = pd.read_csv(data_path, sep=',', index_col=0)
        data = data.T  
        adata = sc.AnnData(X=data.values)
        adata.obs_names = data.index.tolist()  
        adata.var_names = data.columns.tolist()  
    else:
        adata=sc.read_csv(data_path, delimiter=',', first_column_names=True)
    adata.var_names_make_unique()  
    print("Original dataset size:" + str(adata.shape))
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    if args.normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    adata = adata[:, adata.var.highly_variable]
    features=adata.X.astype(np.float32)
    if args.data_type == 'mtx':
        features = features.toarray()
    features=torch.from_numpy(features)
    print("Preprocessed dataset size:" + str(adata.shape))

    #getGraph
    adj,W_NE = getGraph(features,10)

    #scale
    sc.pp.scale(adata, max_value=10)


    #PCA
    sc.tl.pca(adata, svd_solver='arpack')
    #compute neighbors
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    #clustering
    if args.n_clusters == 0:
        sc.tl.leiden(adata)
        num_clusters = len(np.unique(adata.obs['leiden']))
    else:    
        num_clusters = args.n_clusters
    
    
    print("Number of clusters: ", num_clusters)
    cells=list(adata.obs.index)
    print("Number of cells: ", len(cells))

    features=torch.FloatTensor(adata.obsm['X_pca'].copy())

    
    return adj,features,cells,num_clusters
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn

def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix    
def normalization(X):
    # data normalization
    X =np.array(X)
    for i in range(len(X)):
        X[i] = X[i] / sum(X[i]) * 100000
    X = np.log2(X + 1)
    return X

def getGraph(X,K):
    # Construct cell graph
    co_matrix = np.corrcoef(X)
    X = normalization(X)
    in_matrix = np.corrcoef(X)
    
    NE_matrix = getNeMatrix(in_matrix)
    

    data = NE_matrix.reshape(-1)
    data = np.sort(data)
    data = data[:-int(len(data) * 0.02)]

    min_sh = data[0]
    max_sh = data[-1]
    
    delta = (max_sh - min_sh) / 100
    
    temp_cnt = []
    for i in range(20):
        s_sh = min_sh + delta * i
        e_sh = s_sh + delta
        temp_data = data[data > s_sh]
        temp_data = temp_data[temp_data < e_sh]
        temp_cnt.append([(s_sh + e_sh) / 2, len(temp_data)])
    
    candi_sh = -1
    for i in range(len(temp_cnt)):
        pear_sh, pear_cnt = temp_cnt[i]
        if 0 < i < len(temp_cnt) - 1:
            if pear_cnt < temp_cnt[i + 1][1] and pear_cnt < temp_cnt[i - 1][1]:
                candi_sh = pear_sh
                break
    if candi_sh < 0:
        for i in range(1, len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if pear_cnt * 2 < temp_cnt[i - 1][1]:
                candi_sh = pear_sh
    
    if candi_sh == -1:
        candi_sh = 0.3
    
    propor = len(NE_matrix[NE_matrix <= candi_sh]) / (len(NE_matrix) ** 2)
    propor = 1 - propor
    thres = np.sort(NE_matrix)[:, -int(len(NE_matrix) * propor)]
    co_matrix.T[NE_matrix.T <= thres] = 0

    up_K = np.sort(co_matrix)[:, -K]
    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1
    mat_K = (mat_K+mat_K.T)/2
    mat_K[mat_K>=0.5] = 1
   
    W_NE = mat_K*co_matrix
    
    sparse_mat_K = sp.csr_matrix(mat_K)
    return sparse_mat_K,W_NE

def getNeMatrix(W_in):
    N = len(W_in)
    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W




def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W
#Updated
def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        if ~ismember([idx_i,idx_j],edges_all) and ~ismember([idx_j,idx_i],edges_all):
            val_edges_false.append([idx_i, idx_j])
        else:
            # Debug
            print(str(idx_i)+" "+str(idx_j))
        # Original:
        # val_edges_false.append([idx_i, idx_j])

    #TODO: temporary disable for ismember function may require huge memory.
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sparse_mx.tocoo().astype(np.float64)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.DoubleTensor(indices, values, shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])
    
    preds_all = np.hstack([preds, preds_neg])
    
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def saveClusterResult(y_pred, cells, dataset_str):
    save_path = 'data/'+dataset_str+'/pred_labels.csv'
    y_pred=pd.Series(y_pred,index=cells)
    
    try:
        y_pred = y_pred.astype('category')
    except Exception as e:
        print(f"Error converting to category: {e}")
    
    pred_result = pd.DataFrame(y_pred, index=cells, columns=['label'])
    pred_result.to_csv(save_path, index=True)

    