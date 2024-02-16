import re
import json
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from numpy import *  # to override the math functions
from srkit.srkit.primitives import primitives, Primitive

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CharDataset(Dataset):
    def __init__(self, data, block_size, chars,
                 numVars, numYs, numPoints, target='Eqarray', addVars=False):
        data_size, vocab_size = len(data), 23
        print('data has %d examples, %d unique.' % (data_size, vocab_size))

        primitive2token = {}
        token2primitive = {}
        for a, x in enumerate(primitives):
            primitive2token[x.name] = a
            token2primitive[a] = x

        self.stoi = primitive2token
        self.itos = token2primitive
        print(self.stoi)
        print(self.itos)

        self.numVars = numVars
        self.numYs = numYs
        self.numPoints = numPoints

        # padding token
        self.paddingToken = '_'
        self.paddingID = 22
        self.stoi[self.paddingToken] = self.paddingID
        self.itos[self.paddingID] = self.paddingToken
        self.threshold = [-100, 100]

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data  # it should be a list of examples
        self.target = target
        self.addVars = addVars

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # grab an example from the data
        chunk = self.data[idx]  # sequence of tokens including x, y, eq, etc.

        try:
            chunk = json.loads(chunk)  # convert the sequence tokens to a dictionary
        except:
            print("Couldn't convert to json: {}".format(chunk))

        # convert equation to list, add padding around orders to ensure all orders are same length
        eqArray = chunk['Eqarray']

        # extract points from the input sequence
        points = torch.zeros(self.numVars + self.numYs, self.numPoints)
        for idx, xy in enumerate(zip(chunk['X'], chunk['Y'])):
            x = xy[0]
            x = x + [0] * (max(self.numVars - len(x), 0))  # padding

            y = [xy[1]] if type(xy[1]) == float else xy[1]
            y = y + [0] * (max(self.numYs - len(y), 0))  # padding
            p = x + y  # because it is only one point
            p = torch.tensor(p)
            # replace nan and inf
            p = torch.nan_to_num(p, nan=self.threshold[1],
                                 posinf=self.threshold[1],
                                 neginf=self.threshold[0])
            points[:, idx] = p

        points = torch.nan_to_num(points, nan=self.threshold[1],
                                  posinf=self.threshold[1],
                                  neginf=self.threshold[0])

        eqArray = torch.tensor(eqArray, dtype=torch.long)
        return points, eqArray



def processDataFiles(files):
    text = ''""
    for f in tqdm(files):
        with open(f, 'r') as h:
            lines = h.read()  # don't worry we won't run out of file handles
            if lines[-1] == -1:
                lines = lines[:-1]
            text += lines  # json.loads(line)
    return text
