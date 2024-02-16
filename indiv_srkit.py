import torch
from model_srkit import GPT

class Individual:

    def __init__(self, mconf, pconf, device):
        self.encoder = GPT(mconf, pconf)
        self.encoder.to(device)
        self.sym_fitness = torch.inf
        self.num_fitness = torch.inf
        self.sd = self.encoder.state_dict()
