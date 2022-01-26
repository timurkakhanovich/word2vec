import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.central_emb = nn.Embedding(num_embeddings=self.vocab_size, 
                                        embedding_dim=self.embedding_dim)
        self.context_emb = nn.Embedding(num_embeddings=self.vocab_size, 
                                        embedding_dim=self.embedding_dim)

    def forward(self, central, context, neg_samples):
        emb_central = self.central_emb(central)
        emb_context = self.context_emb(context)
        emb_neg_samples = self.context_emb(neg_samples)

        central_out = torch.matmul(emb_context.T, emb_central)
        rest_out = torch.bmm(emb_neg_samples, emb_central.unsqueeze(-1)).squeeze(-1)
        
        return -torch.mean(F.logsigmoid(central_out) + torch.sum(F.logsigmoid(-rest_out)))