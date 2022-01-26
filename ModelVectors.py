import pickle

import torch
import torch.nn.functional as F

class ModelVectors():
    def __init__(self, embedding_path):
        with open('vocab.pkl', 'rb') as fin:
            self.vocab = pickle.load(fin)
        
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.vectors = torch.load(embedding_path)
        self.emb_dim = self.vectors.size(1)

    def __getitem__(self, token):
        token_idx = self.word2idx.get(token, None)

        if token_idx:
            return self.vectors[token_idx]
        else:
            return None
    
    def most_similar(self, positive, negative, top_k=5):
        for pos in positive:
            if pos not in self.word2idx:
                raise Exception('Tokens must be in the vocabulary')
        
        for neg in negative:
            if neg not in self.word2idx:
                raise Exception('Tokens must be in the vocabulary')

        assert len(positive) != 0, 'There must be at least one word in variable positive'

        if len(negative) != 0:
            negatives = torch.stack([self[neg] for neg in negative])
        else:
            negatives = torch.zeros((1, self.emb_dim))
        positives = torch.stack([self[pos] for pos in positive])

        result_vec = torch.sum(positives, dim=0) - torch.sum(negatives, dim=0)
        result_vec = result_vec.unsqueeze(0)

        not_to_consider_indices = [self.word2idx[w] for w in positive + negative]
        consider_indices = [i for i in range(len(self.vocab)) if i not in not_to_consider_indices]

        cosine_distances = torch.topk(
            F.cosine_similarity(result_vec, self.vectors[consider_indices])
        , top_k)

        return (
            [self.vocab[idx] for idx in cosine_distances.indices], 
            cosine_distances.values
        )