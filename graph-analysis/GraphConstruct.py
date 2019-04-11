import numpy as np
import pandas as pd
import warnings
import sys
import networkx as nx
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.neighbors import kneighbors_graph as knngraph
from matplotlib.ticker import MaxNLocator
from Utilities import Utilities as utils
import matplotlib.pyplot as plt
import Gower as gow
from sklearn.neighbors import DistanceMetric
from scipy.sparse import csr_matrix

class GraphConstruct:
    def __init__(self):
        # Does nothing (for now)
        print "Initalized object"
    
    def construct(self, domain, signal, feat_ids = None, method = "knn", k_neighb = 50, \
                  distance = 'Gower', noise=0., subd_sample_idx=[]):
        '''
            Construct neighborhood graph of the underlying data domain using specified 
            methods
            Input:
                domain: data domain of shape (#samples, #parameters) (Pandas DataFrame)
                signal: function value corresponding to each sample/node in the graph - typically
                        the response variable in the data
                feat_ids (optional): Features to be used for constructing the neighborhood graph. If 
                unspecified, use all the features of domain
                method (optional): Technique used to construct the graph; default - knn (only knn
                implemented for now)
                k_neighb: Number of neighbors to be used for constructing the knn graph. Default = 50.
                distance: Distance function to be used. Available options are Manhattan, Euclidean and Gower.
                subd_sample_idx: Indices of samples to be used for the subdomain analysis. Default empty.
            Output:
                G: Neighborhood graph 
        '''
        m_d, n_d = domain.shape
        m_s = signal.shape
        
        assert m_d == m_s[0], "Domain and signal are not of the same size"

        colnames = domain.columns.values.tolist() # Get column headers
        if feat_ids is None:
            feat_ids = np.arange(n_d) 
        features = [colnames[i] for i in feat_ids]
        
        # print "Number of features used = ", len(features)
        
        if method == "knn":
            dist_list = ['Manhattan', 'Euclidean', 'Gower']
            assert distance in dist_list, "Only Manhattan, Euclidean and Gower distances are supported as of now"
            p = dist_list.index(distance) + 1

            if p < 3:
                adj = knngraph(domain[features], n_neighbors=k_neighb, mode='distance', p=p)
                adj = adj.toarray()
                adj += noise*np.random.rand(adj.shape[0], adj.shape[0])
                adj = csr_matrix(adj) # Make it sparse - (shouldn't matter though)
            else: # Gower distance
                # Not doing subdomain analysis
                # Second condition is to check if all the features are being used
                if len(subd_sample_idx) == 0 or len(feat_ids) == n_d:
                    adj = self.call_gower2(domain[features], n_neighbors=k_neighb, noise=noise)
                else: # Perform subdomain analysis
                    # Extract features to be dropped (using the list of features to be used)
                    all_feats = np.arange(n_d)
                    feats_drop = [x for x in all_feats if x not in feat_ids]
                    adj = self.call_gower_subdomain(domain, n_neighbors=k_neighb, noise=noise, feats_drop=feats_drop, \
                                                    subd_sample_idx=subd_sample_idx)
            
            G = nx.Graph(adj)
            #print G.adj
            
        # elif method == ... # To be extended
        
        return G



    def call_gower2(self, data_sample, n_neighbors=50, noise=0.):
        '''
            Adapted from: https://github.com/matchado/Misc/blob/master/gower_dist.py
            
            This function expects a Pandas DataFrame as input.
            The data frame is to contain the features along the columns. Based on these features, a
            distance matrix will be returned containing the pairwise Gower distance between the rows.
        '''
        individual_variable_distances = []

        for i in range(data_sample.shape[1]):
            feature = data_sample.iloc[:,[i]]
            if feature.dtypes[0] == np.object:
                feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
            else:
                feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
                
            individual_variable_distances.append(feature_dist)

        D = np.array(individual_variable_distances).mean(0)
        D += noise*np.random.rand(D.shape[0], D.shape[0])
        
        # Matrix should contain only the closest neighbors values - others can be changed to infinity
        for i in range(len(D)):
            sort_idx = np.argsort(D[i])
            for j in range(len(D[i])):
                if (j not in sort_idx[:n_neighbors+1]) or (j == i):
                    D[i][j] = np.inf
 
        D[D == np.inf] = 0. # To be modified in the future
        D_symm = np.maximum(D, D.T) # Force the matrix to be symmetric
        
        #print "Adj matrix min max", D_symm.min(), D_symm.max()
        D_sparse = csr_matrix(D_symm) # Make it sparse
        return D_sparse


    def call_gower_subdomain(self, data_sample, n_neighbors=50, noise=0., feats_drop=[], subd_sample_idx=[]):

        individual_variable_distances = []
        for i in range(data_sample.shape[1]):
            feature = data_sample.iloc[:,[i]]
            if feature.dtypes[0] == np.object:
                feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
            else:
                feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
        
            # If feature has to be dropped in subdomain
            if i in feats_drop:
                for j in subd_sample_idx:
                    feature_dist[j] = 0.
                    
                    # For samples not in the subdomain, make the adj matrix entries corresponding
                    # to samples in the subdomain 0
                    feature_dist[:, j] = 0.

            individual_variable_distances.append(feature_dist)
        
        D = np.array(individual_variable_distances).mean(0)
        #print "Original \n", D
        D_new = D.copy()
        D_sum = np.array(individual_variable_distances).sum(0)
        
        n_divide = data_sample.shape[1] - len(feats_drop)
        print "Num feats in subdomain = ", n_divide
        for i in subd_sample_idx:
            D_new[i] = (D_sum[i]/n_divide).astype(float)
            D_new[:, i] = (D_sum[:, i]/n_divide).astype(float)
    
        #print "Modified \n", D_new
        #print "Old and new sums", np.sum(D),
        D = D_new.copy()
        #print np.sum(D)
        D += noise*np.random.rand(D.shape[0], D.shape[0])
        
        
        # Matrix should contain only the closest neighbors values - others can be changed to infinity
        for i in range(len(D)):
            sort_idx = np.argsort(D[i])
            for j in range(len(D[i])):
                if (j not in sort_idx[:n_neighbors+1]) or (j == i):
                    D[i][j] = np.inf

        D[D == np.inf] = 0. # To be modified in the future
        D_symm = np.maximum(D, D.T) # Force the matrix to be symmetric
        
        #print "Adj matrix", D_symm
        #print "Adj matrix min max", D_symm.min(), D_symm.max(), "\n\n"
        
        D_sparse = csr_matrix(D_symm) # Make it sparse
        return D_sparse
