from MultiSimNeNc import MultiSimNeNc
import numpy as np
import networkx as nx
from igraph import *
import scipy.io as sio

def read_gml_file_to_graph(gml_file,node_label,weight_lables,true_lables):
    G = nx.read_gml(gml_file, label=node_label)

    adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None, weight=weight_lables).todense())
    adj_matrix[np.eye(len(adj_matrix), dtype=bool)] = 0

    true_lables_dist={}
    for i in range(len(G.adj)):
        if G.node[i][true_lables]:
            true_lables_dist[i] = G.node[i][true_lables]
        else:
            true_lables_dist[i]=0

    true_lables_list = list(true_lables_dist.values())

    nique_lable = set(true_lables_list)
    unique_lable_dict = {item: val for val, item in enumerate(nique_lable)}
    y_true_list = []
    for node in true_lables_list:
        y_true_list.append(unique_lable_dict[node])

    return adj_matrix,y_true_list


def read_edgeList_file_to_graph(edgefile_file,true_lables_file):
    G = nx.read_weighted_edgelist(edgefile_file)

    adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None, weight="weight").todense())

    labels = sio.loadmat(true_lables_file)
    labels = labels['wine_label']
    y_true_lables = []
    for i in labels:
        y_true_lables.append(i[0])

    return adj_matrix,y_true_lables

#Data 1: karate Network
gml_file="./DATA/karate_club.gml"
adj_matrix, true_lables_list = read_gml_file_to_graph(gml_file, "id","weight","club")

#Data 2: Polbooks Network
# gml_file="./DATA/polbooks.gml"
# g,adj_matrix, true_lables_list = read_gml_file_to_graph(gml_file, "id",None,"value")

#Data 3: Wine Network
# gml_file="./DATA/wine.edgelist"
# true_lables_file="./DATA_me/wine_label.mat"
# adj_matrix, true_lables_list=read_edgeList_file_to_graph(gml_file,true_lables_file)
# print(adj_matrix.shape)


print("Actual number of modules:",len(set(true_lables_list)))
print("\n")
print("-------------------------------------RUN MultiSimNeNc------------------------------")

model=MultiSimNeNc(adj_matrix,2,32,0,5)
Best_N,GMM_pre_lables=model.fit()
print("\n")
print("Predicted optimal number of modules:",Best_N)
print("Node label (module identification result):\n",GMM_pre_lables)
