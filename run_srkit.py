import torch
from model_srkit import GPT, GPTConfig, PointNetConfig
from utils_srkit import processDataFiles, CharDataset
import sys
import glob
import random
import numpy as np
from evolve_srkit import evolve

uniqueID = int(sys.argv[1])
# numGenerations = int(sys.argv[2])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
numGenerations = 8000
parentSize = 15
childSize = 15
mutationRate = 0.5
xoverRate = 0.5

embeddingSize = 512  # the hidden dimension of the representation of both GPT and PT
numPoints = 30  # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars = 1  # the dimenstion of input points x, if you don't know then use the maximum
numYs = 1  # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 30  # spatial extent of the model for its context
batchSize = 32  # batch size of training data
dataDir = './datasets/'
target = 'Skeleton'  # 'Skeleton' #'EQ'
dataFolder = 'base'
addr = './SavedModels/'  # where to save model
method = 'EMB_SUM'  # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation.
variableEmbedding = 'NOT_VAR'  # NOT_VAR/LEA_EMB/STR_VAR
addVars = False
maxNumFiles = 30  # maximum number of file to load in memory for training the neural network
bestLoss = None  # if there is any model to load as pre-trained one

#LOAD DATA
path = 'Dataset/100fullop.json'
files = glob.glob(path)[:maxNumFiles]
text = processDataFiles(files)
chars = sorted(list(set(text)) + ['_', 'T', '<', '>',
                                  ':'])  # extract unique characters from the text before converting the text to a list, # T is for the test data
text = text.split('\n')  # convert the raw text to a set of examples
text = text[:-1] if len(text[-1]) == 0 else text
random.shuffle(text)  # shuffle the dataset, it's important specailly for the combined number of variables experiment
train_dataset = CharDataset(text, blockSize, chars, numVars=numVars,
                            numYs=numYs, numPoints=numPoints, target=target, addVars=addVars)
#create models
#encoder
pconf = PointNetConfig(embeddingSize=embeddingSize,
                       numberofPoints=numPoints,
                       numberofVars=numVars,
                       numberofYs=numYs,
                       method=method,
                       variableEmbedding=variableEmbedding)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize,
                  padding_idx=train_dataset.paddingID)
print("------------------------------------------------------------------------")
print("------------------------------------------------------------------------")
print("------------------------------------------------------------------------")


#evolve
evolve(mconf, pconf, train_dataset, device, generations=numGenerations, num_parents=parentSize,
       num_children=childSize, mutation_rate=mutationRate, cross_rate=xoverRate, uniqueID=uniqueID,
       batch_size=batchSize)
