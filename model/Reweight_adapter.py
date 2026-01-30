"Prompt Adapter"
from functools import reduce
from operator import add
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.nn import BatchNorm2d as BatchNorm
from model.Transformer_decoder import transformer_decoder
  
class Reweight_adapter(nn.Module):
    def __init__(self, args):
        super(Reweight_adapter, self).__init__()
        
        hidden_dim = 256
        self.transformer_decoder = transformer_decoder(args, hidden_dim, hidden_dim*2)
    
    def forward(self, image_embedding, prompt_embeddings, attn_mask):

        adjusted_prompts, attn_weights = self.transformer_decoder(image_embedding, prompt_embeddings, attn_mask = None)
        return adjusted_prompts, attn_weights