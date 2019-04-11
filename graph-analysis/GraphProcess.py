import numpy as np
import pandas as pd
import warnings
import sys
import networkx as nx
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.neighbors import kneighbors_graph as knngraph
from matplotlib.ticker import MaxNLocator
from Utilities import Utilities as utils
from GraphConstruct import GraphConstruct
import matplotlib.pyplot as plt
import scipy as sc
from timeit import default_timer as timer
from collections import defaultdict
plt.style.use('seaborn-colorblind')

class GraphProcess:
    def __init__(self, G=None):
        '''
            Create a new class object and initialize its graph with argument G. Also
            create an object to construct graphs given data domain and signal
            Input:
                G (optional) = a networkx Graph object
        '''
        if G is not None:
            self.Graph = G
        else:
            self.Graph = nx.Graph()
        self.gc = GraphConstruct()
    
    def gft_sensitivity(self, domain, signal, feat_ids = None, method = "knn", k_neighb = 50, \
                        distance = "Gower", noise=0., subd_sample_idx=[]):
        '''
            Compute the Fourier spectrum of the input signal
            Input:
                signal: function values of the nodes in the graph - typically the response
                variable in the data
            Output:
                Spectrum of the signal (project the signal onto its Fourier basis) 
                and the inverse of the eigenvector matrix
        '''
        G = self.gc.construct(domain, signal, feat_ids, method, k_neighb, distance, noise, subd_sample_idx)
        signal = minmax_scale(signal) # Standardize the signal
        
        # Compute the eigenvalues and vectors of the (normalized) Laplacian matrix of G
        L = nx.normalized_laplacian_matrix(G)
        e, u = np.linalg.eigh(L.todense())
        
        #print "Laplacian sum, min, max", L.sum(), L.min(), L.max(),
        #print "Eigenval sum min, max", e.sum(), e.min(), e.max()

        # Sort the eigenvector matrix based on the corresponding eigenvalues and 
        # project the signal onto the eigen basis
        idx = np.argsort(e)
        
        # u = np.real(u)
        U = sc.linalg.inv(u[:, idx]) # Scipy returns np ndarray by default
        # U = np.real(U) # Convert to real

        e_sort = e[idx]
        xf = U.dot(signal)
        
        return xf, U
    
    
    # Does the same thing as gft_sensitivity, but also returns the networkx graph G
    # This has been written as a separate function so as to not change every code snippet that uses gft_sensitivity
    def sample_influence_gft_sensitivity(self, domain, signal, feat_ids = None, method = "knn", k_neighb = 50, \
                        distance = "Gower", noise=0., subd_sample_idx=[]):
        G = self.gc.construct(domain, signal, feat_ids, method, k_neighb, distance, noise, subd_sample_idx)
        signal = minmax_scale(signal) # Standardize the signal
        
        # Compute the eigenvalues and vectors of the (normalized) Laplacian matrix of G
        L = nx.normalized_laplacian_matrix(G)
        e, u = np.linalg.eigh(L.todense())
        
        
        # Sort the eigenvector matrix based on the corresponding eigenvalues and
        # project the signal onto the eigen basis
        idx = np.argsort(e)
        
        U = sc.linalg.inv(u[:, idx]) # Scipy returns np ndarray by default
        
        e_sort = e[idx]
        xf = U.dot(signal)
        
        return xf, U, G

    
    
    def sensitivity_vector(self, domain, signal, feat_ids = None, method = 'knn', k_neighb = 50,\
                          distance = 'Gower', freq=50, drop_together=None, noise=0., subd_sample_idx=[]):
        '''
            Given a set of samples with 'T' parameters, compute the sensitivity of each
            parameter as the distortion induced by removing that parameter.
            
            'freq' determines the frequency after which the energy has to be measured
            drop_together (optional): The list of lists of features that must be considered together as parameters
            to be dropped before computing their influence - typically used for discrete features that have
            been converted into multiple features by binarization of each of their levels. Each sublist will be considered
            as one feature while computing the spectral error. Currently each feature is specified by its index.
            
            Output:
                A vector containing the spectral errors obtained by removing each parameter
        '''
        # Spectrum with all the data in the domain
        x_f, U_orig = self.gft_sensitivity(domain, signal, feat_ids, method, k_neighb, distance, noise, subd_sample_idx)
        spectral_error = []
        
        if feat_ids is None:
            feat_ids = np.arange(domain.shape[1]) 
        
        n_features = len(feat_ids)
        spectral_error = [None]*n_features # Create a list to store the spectral errors

        all_feats = np.arange(n_features)
        feats_left = all_feats.tolist()

        if drop_together is not None:
            for sub_list in drop_together: # Consider each sublist in the list as one feature and drop them together
                feat_ids2 = np.delete(all_feats, sub_list, axis=0)
            
                for elem in sub_list: # Remove the sublist features from the features yet to be processed
                    feats_left.remove(elem)
                print sub_list,

                # Spectrum with one less feature (in this case, a set of features)
                x_hat_f, U_orig = self.gft_sensitivity(domain, signal, feat_ids2, k_neighb=k_neighb, distance=distance, \
                                                       noise=noise, subd_sample_idx=subd_sample_idx)
                
                # Compute spectral error
                curr_error = np.abs(np.linalg.norm(x_f[freq:])**2 - np.linalg.norm(x_hat_f[freq:])**2)

                # Spectral error for each index in the sublist will be the same
                for id in sub_list:
                    spectral_error[id] = curr_error

        # Processed all ids in drop_together, but some features are pending; or drop_together is None:        
        if feats_left:  
            for i in range(n_features):
                if i in feats_left:
                    print i,
                    delid = [i]
                    feat_ids2 = np.delete(all_feats, delid, axis=0)
                    
                    #print "\nFeatures", feat_ids2
                    # Spectrum with one less feature 
                    x_hat_f, U_orig = self.gft_sensitivity(domain, signal, feat_ids2, k_neighb=k_neighb, distance=distance, \
                                                           noise=noise, subd_sample_idx=subd_sample_idx)
                    
                    # Compute spectral error and update the list of spectral errors
                    #print "U_orig sum", np.sum(U_orig),
                    #print "U_orig min max", np.min(U_orig), np.max(U_orig)
                    #print "Norms: A:", np.linalg.norm(x_f[freq:])**2, "; B:", np.linalg.norm(x_hat_f[freq:])**2,
                    curr_error = np.abs(np.linalg.norm(x_f[freq:])**2 - np.linalg.norm(x_hat_f[freq:])**2)
                    spectral_error[i] = curr_error
                
        return np.array(spectral_error)
            
    def draw_graph(self, signal):
        '''
            Draw the graph with a spectral layout (function values of the nodes
            are given by 'signal')
        '''
        nx.draw_spectral(self.Graph, node_color=signal, cmap='bone')
    

    def graph_analysis(self, domain, signal, k_neighb=50, sample_frac=0.01, seed=0, response_name='ExecTime',\
                       feats_drop = None, distance='Gower', freq=50, drop_together = None, noise=0., \
                       num_samples = None, signal_transform = None):
        '''
           Create a random sample of the data and perform graph-based influence analysis on it - Also, plot the feature
           sensitivity graph and the Fourier Spectrum.
           
           Input:
               domain: data domain of shape (num_samples, num_features), doesn't include the response variable
               signal: function value (typically the response variable)
               k_neighb: Number of neighbors to be used for constructing the knn graph. Default = 50.
               sample_frac (optional): Percentage of the data that will be randomly sampled for the analysis. Default = 0.01.
               seed (optional): Random seed for the sampling. Default = 0.
               response_name (optional): Name of the response variable. Default - ExecTime
               feats_drop (optional): Features to be dropped before constructing the neighborhood graph. If
               unspecified, use all the features of domain
               distance (optional): Distance function to be used. Available options are Manhattan, Euclidean and Gower. Default - Gower
               freq (optional): determines the frequency after which the energy has to be measured
               drop_together (optional): The list of lists of features that must be considered together as parameters
               to be dropped before computing their influence - typically used for discrete features that have
               been converted into multiple features by binarization of each of their levels. Each sublist will be considered
               as one feature while computing the spectral error. Currently each feature is specified by its index.
               noise (optional): Measure of the amount of noise to be added to the adjacency matrix. Default - no noise.
               num_samples (optional): Number of data samples to be used from the entire data. If not specified, sample_frac is used to determine the number of samples. If specified, then specified number of samples are used (this overrides sample_frac).
               signal_transform (optional): Dictates whether the signal should be used as is or shoud a transformation be applied on it. Currently available options are 'log' (natural log transformation) and None (use as is). Default - Use as is.
           Output:
               data_sample: The randomly sampled data used for performing the analysis
               spectral_error: 1-D array of spectral errors (obtained by performing graph analysis by removing each feature)
               xf: Fourier spectrum
               U: Eigenvector matrix
        '''
        
        # Create a dataframe and drop unnecessary features
        data = domain
        
        # Should the signal be transformed?
        if signal_transform == 'log':
            signal = np.log(1 + signal)

        data[response_name] = signal
        if feats_drop is not None:
            data = data.drop(feats_drop, axis=1)

        if num_samples is None:
            data_sample = data.sample(frac=sample_frac, random_state=seed)
        else:
            data_sample = data.sample(n=num_samples, random_state=seed)
        print "Data sample shape:", data_sample.shape

        # Compute Feature Sensitivity
        print "Computing feature sensitivities for features:"
        n_feat = data_sample.shape[1] - 1
        spectral_error = self.sensitivity_vector(domain=data_sample.iloc[:, :n_feat], \
                                                    signal=data_sample.iloc[:, n_feat],\
                                                    k_neighb=k_neighb, distance=distance, freq=freq,\
                                                    drop_together=drop_together, noise=noise)
        colnames = data.columns.values
        
        print "\n\nPlotting feature sensitivity - In the order of decreasing feature importance:"
        self.plot_sensitivity(spectral_error, colnames)
        
        print "\n\nComputing and Plotting the Fourier Spectrum"
        xf, U = self.gft_sensitivity(domain=data_sample.iloc[:, :n_feat], \
                                        signal=data_sample.iloc[:, n_feat], \
                                        k_neighb=k_neighb, distance=distance, noise=noise)

        util = utils() # Create a utils object
        util.xy_plot(xf, marker=None, ylabel='Fourier component', title='Fourier Freq Spectrum')
        
        return data_sample, spectral_error, xf, U
    
    def plot_sensitivity(self, spectral_error, ticks):
        '''
            Plot of the spectral errors in descending order
            Input:
                spectral_error: Vector containing the spectral errors corresponding to each feature
                ticks: Names of the features
        '''
        fig, ax = plt.subplots()
        y = np.array(spectral_error)
        args = np.argsort(-y)
        
        ticks = [ticks[idx] for idx in args]
        ax.plot(range(len(args)),y[args],'o-')
        plt.xlabel('Parameter Name')
        plt.ylabel('Approximation Error')
        plt.grid()
        for i, txt in enumerate(ticks):
            ax.annotate(txt, (i,y[args[i]]),rotation=40,horizontalalignment='left',verticalalignment='bottom',)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticklabels([]) # Remove numbers from x axis
        plt.show()


    def subdomain_analysis(self, domain, signal, k_neighb=50, response_name='ExecTime', distance='Gower', \
                       freq=50, drop_together=None, noise=0.01, signal_transform = None, subd_sample_idx=[]):
        '''
            Perform subdomain analysis - Treat k samples from the entire data domain as a subdomain and perform
            LOO parameter graph analysis by removing each parameter only for the subdomain and to any node connected
            to a node in the subdomain
            
            Input:
                domain: data domain of shape (num_samples, num_features), doesn't include the response variable
                signal: function value (typically the response variable)
                k_neighb: Number of neighbors to be used for constructing the knn graph. Default = 50.
                seed (optional): Random seed for the sampling. Default = 0.
                response_name (optional): Name of the response variable. Default - ExecTime
                distance (optional): Distance function to be used. Available options are Manhattan, Euclidean and Gower. Default - Gower
                freq (optional): determines the frequency after which the energy has to be measured. Default = 50.
                drop_together (optional): The list of lists of features that must be considered together as parameters
                to be dropped before computing their influence - typically used for discrete features that have
                been converted into multiple features by binarization of each of their levels. Each sublist will be considered
                as one feature while computing the spectral error. Currently, each feature is specified by its index.
                noise (optional): Measure of the amount of noise to be added to the adjacency matrix. Default = 0.01.
                signal_transform (optional): Dictates whether the signal should be used as is or shoud a transformation be applied on it. Currently available options are 'log' (natural log transformation) and None (use as is). Default - Use as is.
                subd_sample_idx: List of indices of the data domain corresponding to the samples that have be to be treated as the subdomain.
            Output:
                spectral_error_global: 1-D array of spectral errors (obtained by performing graph analysis on the entire data)
                spectral_error_subd: 1-D array of spectral errors (obtained by performing subdomain graph analysis)
                Also plots the spectral errors, the squared and L1 differences between them and prints the change in feature ordering
        '''
    
        # Copy the data
        data = domain
        
        # Should the signal be transformed?
        if signal_transform == 'log':
            signal = np.log(1 + signal)

        data[response_name] = signal
        print "Data head: \n"
        print data.head()
        # Construct adjacency matrix with entire data, perform graph analysis and plot the influences
        print "\n\n-------------------- With the entire data --------------------"
        print "Computing feature sensitivities for features:"
        n_feat = data.shape[1] - 1
        spectral_error_global = self.sensitivity_vector(domain=data.iloc[:, :n_feat], \
                                                          signal=data.iloc[:, n_feat],\
                                                          k_neighb=k_neighb, distance=distance, freq=freq,\
                                                          drop_together=drop_together, noise=noise,\
                                                          subd_sample_idx = [])
            
        colnames = data.columns.values[:n_feat]
      
        print "\n\nPlotting feature sensitivity - In the order of decreasing feature importance:"
        self.plot_sensitivity(spectral_error_global, colnames)
          
        print "\n\n -------------------- Subdomain analysis --------------------"
        print "\n Number of samples in the subdomain: ", len(subd_sample_idx)
        print "\nRemoving feature:"
        spectral_error_subd = self.sensitivity_vector(domain=data.iloc[:, :n_feat], \
                                                        signal=data.iloc[:, n_feat],\
                                                        k_neighb=k_neighb, distance=distance, freq=freq,\
                                                        drop_together=drop_together, noise=noise,\
                                                        subd_sample_idx = subd_sample_idx)
        print "\n\nPlotting feature sensitivity (subdomain analysis) - In the order of decreasing feature importance:"
        self.plot_sensitivity(spectral_error_subd, colnames)
      
        # Plot the L1 and squared spectral error difference
        self.plot_error_diff(spectral_error_global, spectral_error_subd, colnames)

        print "\n\nChange in feature importance order:\n"
        self.feat_ord_change(spectral_error_global, spectral_error_subd, colnames)

        return spectral_error_global, spectral_error_subd


    def feat_ord_change(self, spectral_error_global, spectral_error_subd, colnames):
        '''
            Compute the change in feature influence ordering (between the spectral errors obtained from two
            different graph analysis) - typically used when performing subdomain analysis
            
            Input:
                spectral_error_global: 1-D array of reference spectral errors (typically obtained by performing graph analysis on the entire data)
                spectral_error_subd: 1-D array of spectral errors to be compared against the reference (typically obtained by performing subdomain graph analysis)
                colnames: Names of the features/columns corresponding to the spectral errors
            Output:
                The change in feature influence ordering
        '''
        glob_ord = np.argsort(-spectral_error_global)
        subd_ord = np.argsort(-spectral_error_subd)
        glob_list = glob_ord.tolist()
        subd_list = subd_ord.tolist()
        print '%10s'%('Feature')," Change in order"
        print "---------------------------"
        for i, name in enumerate(colnames):
            print '%10s'%name, "\t\t", glob_list.index(i) - subd_list.index(i)

    def plot_error_diff(self, spectral_error_global, spectral_error_subd, colnames):
        '''
            Plot the L1 and squared difference in spectral errors obtained from two
            different graph analysis - typically used when performing subdomain analysis
            Input:
                spectral_error_global: 1-D array of reference spectral errors (typically obtained by performing graph analysis on the entire data)
                spectral_error_subd: 1-D array of spectral errors to be compared against the reference (typically obtained by performing subdomain graph analysis)
                colnames: Names of the features/columns corresponding to the spectral errors
            Output:
                Plots of the L1 and squared differences
            
        '''
        col_idx = np.arange(len(colnames))
        
        # squared diff
        error_diff_sq = (spectral_error_global - spectral_error_subd)**2
        plt.bar(col_idx, error_diff_sq)
        plt.xticks(col_idx, colnames)
        plt.title('Squared difference in spectral errors')
        plt.show()
        
        # L1 diff - the sign of the change is also important to us?
        error_diff = (spectral_error_global - spectral_error_subd)
        plt.bar(col_idx, error_diff)
        plt.xticks(col_idx, colnames)
        plt.title('L1 difference in spectral errors')
        plt.show()


    def plot_multiple_file_error(self, spectral_errors, dataset_names, colnames):
        '''
            Compute and plot pairwise L1 errors for each feature across datasets
        '''
        sp_vals = spectral_errors.values()
        feat_sp_vals = defaultdict(lambda:[]) # Dict to store the values for each feature

        for i, feat in enumerate(colnames):
            for item in sp_vals:
                feat_sp_vals[i].append(item[i])
        num_datasets = len(colnames)

        # Generate pairwise (cartesian) indices
        pair_indices = zip(np.triu_indices(num_datasets-1)[0], np.triu_indices(num_datasets-1)[1])
        
        # Compute pairwise L1 errors for each feature across datasets
        for feat_id, feat_name in enumerate(colnames):
            l1_errors = np.zeros((num_datasets-1, num_datasets-1)) # Placeholder to store l1 errors
            for i, j in pair_indices:
                l1_errors[i, j] = np.abs(feat_sp_vals[feat_id][i] - feat_sp_vals[feat_id][j])
            l1_errors = np.maximum(l1_errors, l1_errors.T)     # Make matrix symmetric
            plt.imshow(l1_errors)
            plt.colorbar()
            plt.title(feat_name + " - L1 difference across datasets")
            plt.show()


    def graph_analysis_files(self, filenames, parent_folder='../../datasets/powerperf/new-data/', response='ExecTime',\
                         feats_drop=None, k_neighb=50, sample_frac=0.1, num_samples=None, freq=50, distance='Gower', \
                         drop_together=None, noise=0.01, signal_transform=None, \
                         convert_float=['PKG_LIMIT', 'OMP', 'Ranks', 'ExecTime'], \
                         convert_obj=['Nesting', 'Dset', 'Gset']):
        '''
            Perform graph analysis on different files and display/compare the results (frequency spectrum, parameter
            influence plot and differences in parameter influence ranking)
            
            Input:
                filenames: List of file names containing the data to be analyzed
                parent_folder: Path to the parent directory
                convert_float: Parameters to be converted to float
            Output:
                Relevant plots and
                sp_errors: Dict of spectral errors corresponding to different datasets
                xfs: Dict of frequency spectra corresponding to different datasets
        '''
    
        sp_errors, xfs = [{} for i in range(2)]
        util = utils()
        start_overall = timer()
        for i, f in enumerate(filenames):
            filename = parent_folder + f
            X, y = util.load_data(filename, response)
            print "\n--------------------------------------------------------------------------------------------------"
            print "\nGraph Analysis on:", filename
            print "X, y shape: ", X.shape, y.shape # Verify shape

            # Drop unnecessary columns and normalize the data
            data_pd = util.data_norm(X, y, feats_drop=feats_drop, response_name='ExecTime', response_norm=False)

            # Convert datatypes of a few parameters to float
            data_new = util.convert_pd_dtype(data_pd, convert_float, totype='float64')

            start = timer()
            n_feat = data_new.shape[1]-1


            if num_samples is None:
                data_sample, spectral_error, xf, U = self.graph_analysis(data_new.iloc[:, :n_feat], data_new.iloc[:, n_feat],
                                                                    k_neighb=k_neighb, sample_frac=sample_frac, freq=freq, \
                                                                    distance=distance, drop_together=drop_together, \
                                                                    noise=noise, signal_transform=signal_transform)
            else:
                data_sample, spectral_error, xf, U = self.graph_analysis(data_new.iloc[:, :n_feat], data_new.iloc[:, n_feat],
                                                                         k_neighb=k_neighb, num_samples=num_samples, freq=freq, \
                                                                         distance=distance, drop_together=drop_together, \
                                                                         noise=noise, signal_transform=signal_transform)
                        
            sp_errors[i] = spectral_error
            xfs[i] = xf
            end = timer()
            print "\nTime elapsed for current data = ", (end-start), "seconds"
            print "--------------------------------------------------------------------------------------------------\n"

        print "Total time elapsed = ", (timer() - start_overall), "seconds"


        print "\nFeature rankings for different datasets (in the following order)"
        for f in filenames:
            print f

        padding = 10
        i = 0
        colnames = data_sample.columns.values
        for col in colnames[:n_feat]:
            print "\n%s" % col.ljust(padding),
            for item in sp_errors:
                # Sort errors in descending order and print each feature's rank
                sorted_error = np.argsort(-sp_errors[item]).tolist()
                print "%2d " % (sorted_error.index(i)), 
            i += 1
        
        print "\nPlot L1 error matrix for each feature (across datasets)"
        self.plot_multiple_file_error(sp_errors, filenames, colnames[:n_feat])
        print "----------------------- DONE with analysis across datasets using #samples =", num_samples, "------------------"
        print "\n--------------------------------------------------------------------------------------------------\n"
        return sp_errors, xf
