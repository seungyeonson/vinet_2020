# arguments

import argparse
parser = argparse.ArgumentParser()

##paths option
parser.add_argument('-dataset', help='dataset to be used for training the network', default= '')

##Model option
parser.add_argument('-loadModel', help='load pretrained weights, if you want add pretrained weights, give path, else None',default=None)
parser.add_argument('-modelType', help='Type of the model to be loaded:1. vinet_batchnorm 2. vinet 3. our_model', type = str.lower, \
                    choices = ['vinet_batchnorm', 'vinet', 'our_model'],default = None)
parser.add_argument('-activation', help='Actication function to be used', type = str.lower, choices=['relu',''], default= 'relu')

##Dataset option

##Hyperparameter

##experiments, visualization

