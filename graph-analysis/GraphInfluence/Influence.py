import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import kneighbors_graph as knngraph
from sklearn.preprocessing import minmax_scale


def filter_function(adj, f, p=2, type='critic'):
    '''
        Compute influence of samples using a graph signal filtering approach
        Input:
            adj: Adjacency matrix
            f: Graph signal
            p: Number of hops (of neighborhood) used for the filtering, higher ==> stronger filtering; default: p=2
            type: Type of frequency filtering; supports only 'criticisms' for now
    '''
    
    # Construct graph, compute degree matrix and normalized adjacency matrix
    G = nx.Graph(adj)
    degree =  G.degree()
    deg = [1./d[1] for d in degree.items()]
    Dinv = sp.csr_matrix(np.zeros((adj.shape[0],adj.shape[1])))
    idx0,idx1 = np.diag_indices(adj.shape[0])
    Dinv[idx0,idx1] = deg
    A = Dinv*adj
    A_norm = np.sqrt(Dinv)*adj*np.sqrt(Dinv)
    C = adj>0

    P = A_norm**p
    M = sp.csr_matrix.sum(P>0,axis=1)
    

    # (A) Compute a few matrices for preprocessing
    temp = np.ndarray((len(f), len(f)))
    temp2 = np.ndarray((len(f), len(f)))
    for i in range(len(temp)):
        temp[i, :] = f[i]         # ith function value repeated in the entire row i
        temp2[i, :] = f.squeeze() # Entire function repeated in each row


    # (B) Choose influence metric - different choices to experiment with (Metric4 seems to work best)
    #delta = 0.1
    #tmp_new = (temp <= temp2 + delta)  # Metric1: Check if function value <= function value of neighbors + delta
    #tmp_new = temp2                    # Metric2: Directly using the function value of the neighbors
    #tmp_new = np.abs(temp - temp2)     # Metric3: Sum of absolute differences in function values
    tmp_new = f*f.T                     # Metric4: Sum of products of function values at a node and its neighbors
    assert tmp_new.shape == C.shape, 'Check dimensions of function' # To be run only for Metric 4

    
    x = C.multiply(tmp_new)
    tmp0 = np.sum(x.todense(), axis=1, dtype=np.float)
    tmp1 = np.sum(C.todense(), axis=1, dtype=np.float)

    if type=='critic':
        print('Computing function for Criticisms')

        # Use f_0 = 1 - tmp0/tmp1.A if the metric is a measure of trust and you want a measure of distrust
        f_0 = tmp0/tmp1.A
        print np.min(f_0), np.max(f_0)

        f_1 = (P*f_0)/M
        f_filter = f_1-(P*f_1)/M

    assert np.sum(f_0)*np.sum(1-f_0) != 0,'function has constant value 0/1'
    
    # Normalize and return influences
    I = np.abs(f_filter)
    I = I/np.max(I)
    return I.A
