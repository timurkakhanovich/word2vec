import re

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, vocab=None, window_size=5, K_sampling=100):
        """
        Dataset class. Preprocessing data according to vocabulary.  
        :param texts: corpus of texts, 
        :param vocab: vocabulary.  
        """
        
        super().__init__()
        assert window_size % 2 == 1, "Window size must be odd!"
        self.window_size = window_size
        self.K = K_sampling

        self.texts = texts
        
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = set()
            for sample in self.texts:
                self.vocab.update(set(self.sample_preprocess(sample['headline'])))
                self.vocab.update(set(self.sample_preprocess(sample['short_description'])))

            self.vocab = list(self.vocab)
            self.vocab.extend(["<UNK>", "<PAD>"])

        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.UNK, self.PAD = self.word2idx["<UNK>"], self.word2idx["<PAD>"]

        # Preprocessing and splitting text into triplets.  
        self.deuce_sample = list()
        for sample in texts:
            self.deuce_sample.extend(self.split_sentence(
                self.sample_preprocess(sample)
            ))

    def sample_preprocess(self, sample):
        """
        Static method for text preprocessing.  
        :param sample: sample to preprocess.
        :returns: tokenized and preprocessed string -> list.  
        """

        # To lower.  
        sample = sample.lower()

        # Replacing words contructions.  
        contradiction_dict = {"ain't": "are not", "'s": " is", "aren't": "are not", 
                            "i'm": "i am", "'re": " are", "'ve": " have"}
        for word, replaceVal in contradiction_dict.items():
            sample = sample.replace(word, replaceVal)
        
        # Remove hashtags, @users and links.  
        reg_str = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
        sample = re.sub(reg_str, " ", sample)
        
        # Replace numbers with NUM.  
        sample = re.sub('[0-9]+', '<NUM>', sample)

        # Remove multiple spaces.  
        sample = re.sub(' +', ' ', sample)

        sample = sample.strip().split()

        return torch.LongTensor([self.word2idx.get(token, self.UNK) 
                                for token in sample])
    
    def split_sentence(self, sample):
        sent_split = list()
        for token_idx in range(len(sample) - self.window_size + 1):
            central = sample[token_idx + self.window_size//2]
            context_list = sample[token_idx : token_idx+self.window_size]
            context_list = context_list[context_list != central]
            
            for context in context_list:
                sent_split.append([central, context])

        return sent_split

    def __getitem__(self, idx):
        central, context = self.deuce_sample[idx]

        # Negative sampling.  
        while True:
            neg_samples = torch.randint(0, len(self.vocab), (self.K,))
            
            if (central not in neg_samples) and (context not in neg_samples):
                break
        
        return central, context, neg_samples
    
    def __len__(self):
        return len(self.texts)

def collate_fn(batch):
    central, context, negatives = list(), list(), list()
    for b in batch:
        central.append(b[0])
        context.append(b[1])
        negatives.append(b[2])
    
    central = torch.stack(central)
    context = torch.stack(context)
    negatives = torch.stack(negatives)

    return central, context, negatives