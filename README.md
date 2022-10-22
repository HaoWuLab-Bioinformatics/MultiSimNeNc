# MultiSimNeNc
MultiSimNeNc is a novel network-based method for identifying modules in various types of networks by integrating network representation learning (NRL) and clustering algorithms.

## Introduction
MultiSimNeNc integrate the two fields of network representation learning (NRL) and data clustering. This method can capture the multi-order similarity of different types of networks using graph convolution (GC) and integrate this similarity information to enhance the modular structure of the network, and also find the optimal number of modules using the Bayesian Information Criterion (BIC) to overcome the shortcoming that most module identification algorithms require the prior specification of the number of modules.

![image](https://github.com/HaoWuLab-Bioinformatics/MusimNeNc/blob/main/flowchat.jpg)

## Overview
The DATA dataset contains four network.where Karate and Pollbooks are unweighted graphs; Wine and Butterfly are fully weighted graphs.

The MultiSimNeNc.py is the proposed model of this paper. we have encapsulated it as a class, which can be called directly. However, it is necessary to understand the meaning of the parameters and to set the appropriate functions for the different networks. 
In particular, the parameter dim must not be larger than the number of network nodes.

The RUN_MultiSimNeNC.py is a sample program.Can be run directly.

## Dependency
python 3.8  
numpy 1.22
networkx  2.3
scipy  1.8.0
scikit-learn  1.0.2

## Usage
` python RUN_MultiSimNeNC.py `
Module identification


