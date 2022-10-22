
import pandas as pd
from sklearn import preprocessing
from scipy.stats import kendalltau
from igraph import *
from sklearn.decomposition import NMF
from sklearn import mixture
from igraph import *
import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from munkres import Munkres


class MultiSimNeNc():
    """
    An implementation of `"MultiSimNeNc" from the papar "MultiSimNeNc: network representation learning-based module identification algorithm by network embedding and clustering"
    The procedure uses graph convolution on a network adjacency matrix to obtain multi-order similarity and uses BIC to find the optimal number of modules to identify modules.

    Args:
        adj_matrix(np.asarray):The adjacency matrix of the network.
        K(int): Set the similarity information for calculating K steps.
        dim(int):The target dimension of NMF dimensionality reduction.
        KNN(int): Set the network information used when different types of networks perform graph convolution.
                   The default value is 0, which means that the full image information is used for graph convolution;
                   when it is a Complete Graph, it is set to 20 in this study.
        max_N(int):Predict the maximum number of modules.
    """

    def __init__(self, adj_matrix,K,dim,KNN, max_N):
        self.adj_matrix = adj_matrix
        self.K=K
        self.shape=adj_matrix.shape
        self.dim=dim
        self.KNN_K=KNN
        self.max_N=max_N

    def fit(self):
        """
        run the model
        :return: Returns the module identification result
        """

        self.SA=self.calculate_SA() #The adjacency similarity matrix
        self.S_K=self.graph_convolution_S_K() #Convolution to extract similarity information
        self.S = self.Similarity_Matrix_Integration() #Similarity information integration
        self.nodeDate=self.to_NMF() #NMF dimensionality reduction
        preLables=self.node_cluster(2,self.max_N) #Module identification
        return preLables


    def  calculate_SA(self):
         """
         Uses the linear proportional transformation method to normalize the adjacency matrix A.

         :return:SA(The adjacency similarity matrix)
         """
         self.adj_matrix[np.eye(len(self.adj_matrix), dtype=bool)] = 0
         SA = preprocessing.normalize(self.adj_matrix, norm='max', axis=1)
         SA[np.eye(len(SA), dtype=bool)] = 1
         return SA

    def k_nearest_neighbors(self):
        """
        Calculate the KNN adjacency matrix

        :return:KNN adjacency matrix
        """

        self.adj_matrix[np.eye(len(self.adj_matrix), dtype=bool)] = 0
        aff_matrix_K = []
        n = self.adj_matrix.shape[0]

        for i in range(n):
            list_i = self.adj_matrix[i]
            dict_nei_i = {}
            for j, w_ij in enumerate(list_i):
                dict_nei_i[
                    j] = w_ij

            sort_dict_nei_i = sorted(dict_nei_i.items(), key=lambda d: d[1], reverse=True)
            KNN_list = sort_dict_nei_i[0:self.KNN_K]
            KNN_Ni = []
            for i_k in KNN_list:
                KNN_Ni.append(i_k[0])
            New_list_i = np.zeros(len(list_i))
            for j in range(len(list_i)):
                if j in KNN_Ni:
                    New_list_i[j] = list_i[j]
                else:
                    New_list_i[j] = 0
            aff_matrix_K.append(New_list_i)
        aff_matrix_K = np.array(aff_matrix_K)
        return aff_matrix_K

    def graph_convolution_A_K(self):
        """
        Perform a graph convolution operation to calculate the feature matrix for each convolutional layer.

        :return:Feature matrix on each convolutional layer
        """

        if self.KNN_K == 0:
            convolve_matrix = self.adj_matrix
        else:
            # 针对完全图，若直接进行全图卷积，则会导致网络结构信息获取不佳，此时可以利用其KNN矩阵来进行图卷积获取结构信息
            convolve_matrix = self.k_nearest_neighbors()

        convolve_matrix[np.eye(len(convolve_matrix), dtype=bool)] = 1
        n, m = convolve_matrix.shape
        t = 0
        g = np.identity(n)
        convolution_result = []
        while t < self.K:
            g = self.graph_convolution_updata(convolve_matrix, g)
            g=self.graph_convolution_diag(g,convolve_matrix)
            convolution_result.append(g)
            t = t + 1
        return convolution_result

    def graph_convolution_diag(self,g,convolve_matrix):
        """
        Feature update of its own node on each convolutional layer
        """
        n, m = convolve_matrix.shape
        g = np.copy(g)
        g[np.eye(m, dtype=bool)] = 0
        for i in range(n):
            for j in range(m):
                if i == j:
                    list_i = convolve_matrix[i]
                    nei_x_list = np.nonzero(list_i)[0]
                    W_x = 0
                    for k in nei_x_list:
                        W_x = g[i, k] + W_x
                    W_x = W_x + 1
                    g[i, j] = W_x
        return g

    def graph_convolution_updata(self,M,g_t):
        """
        Feature update at each convolutional layer
        """
        n, m = g_t.shape
        g_new = []
        for x in range(n):
            adj_X = M[x]
            nei_x_list = np.nonzero(adj_X)[0]
            W_x = []
            for i in nei_x_list:
                W_x_i = M[x, i] * g_t[i]
                W_x.append(W_x_i)
            W_x = np.array(W_x)
            g_new_x_i = 1 * W_x.sum(axis=0)
            g_new_x = g_t[x] + g_new_x_i
            g_new.append(g_new_x)
        g_new = np.array(g_new)
        return g_new

    def graph_convolution_S_K(self):
        """
        Calculate similarity information from the feature matrix on each convolutional layer
        :return:S_K (all similar information)
        """
        convolution_result = self.graph_convolution_A_K()
        S_K = []
        for i, convolution_matrix in enumerate(convolution_result):
            S = self.similar_in_structure(convolution_matrix)
            S_K.append(S)
        return S_K

    def similar_in_structure(self,convolution_matrix):
        """
        Calculate the similarity matrix
        :param convolution_matrix: Feature matrix on convolutional layers
        :return:similarity matrix
        """

        n, m = self.shape
        S_matrix = np.empty((m, m))
        for i, j in zip(*np.triu_indices(m, 1)):
            x = np.copy(convolution_matrix[:, i])
            y = np.copy(convolution_matrix[:, j])
            x[i] = 0
            x[j] = 0
            y[i] = 0
            y[j] = 0
            nonzeroind_X = np.nonzero(x)[0]  # the return is a little funny so I use the [0]
            nonzeroind_y = np.nonzero(y)[0]
            if list(set(nonzeroind_X) & set(nonzeroind_y)):
                if (np.sum(x) == 0) & (np.sum(y) == 0):
                    p_r, p_p = kendalltau(convolution_matrix[:, i], convolution_matrix[:, j])
                else:
                    p_r, p_p = kendalltau(x, y)
            else:
                p_r = 0

            S_matrix[i, j] = p_r
            S_matrix[j, i] = p_r
        S_matrix[S_matrix < 0] = 0
        S_matrix[np.eye(len(S_matrix), dtype=bool)] = 1
        return S_matrix

    def Similarity_Matrix_Integration(self):
        """
        form the total similarity matrix S
        :return: S
        """
        S=self.SA
        S[np.eye(len(S), dtype=bool)] =1
        for i, S_i in enumerate(self.S_K):
            S = S + 1 * S_i
        S_inf = np.isnan(S)
        S[S_inf] = 0
        return S

    def to_NMF(self):
        """
        Dimensionality reduction of the total similarity matrix S by NMF
        :return: nodeDate(low-dimensional node data)
        """
        model = NMF(n_components=self.dim,init="nndsvdar",beta_loss='frobenius',solver="mu",max_iter=2000)
        nodeDate = model.fit_transform(self.S)
        return nodeDate

    def node_cluster(self,min_N,max_N):
        """
        Node clustering
        :param min_N: Minimum number of modules
        :param max_N: Maximum number of modules
        :return: Module identification result
        """
        lowest_bic = np.infty
        Bic = []
        best_n_component = 2
        best_module = mixture.GaussianMixture(n_components=best_n_component)
        n_components_range = range(min_N, max_N)
        cv_types = ["full","tied"]
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                GMM= mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
                GMM.fit(self.nodeDate)
                bic_N = GMM.bic(self.nodeDate)

                Bic.append(bic_N)

                if Bic[-1] < lowest_bic:
                    lowest_bic = Bic[-1]
                    best_n_component = n_components
                    best_module=GMM

        preLables = best_module.predict(self.nodeDate)

        return best_n_component,preLables









