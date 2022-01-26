from base64 import encode
import mlflow
import sys
import pickle

import torch
from torch.utils.data import DataLoader

from dataset import TextDataset, collate_fn

from train import train
from model_config import ModelConfig

def main_command_line_args():
    CHECKPOINT_PATH = "Checkpoints/"
    WINDOW = 5

    configs = ModelConfig(CHECKPOINT_PATH).get_arguments()

    with open('dataset.txt', 'rb') as fout:
        texts = pickle.load(fout)

    with open('vocab.pkl', 'rb') as fout:
        vocab = pickle.load(fout)

    device = configs['device']
    model = configs['model']
    optimizer = configs['optimizer']
    scheduler = configs['scheduler']
    num_epochs = configs['num_epochs']
    start_epoch = configs['start_epoch']
    history = configs['history']
    lr = configs['lr']
    K = configs['K']
    batch_size = configs['batch_size']
    
    text_dataset = TextDataset(texts, vocab=vocab, window_size=WINDOW, K_sampling=K)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    mlflow.set_tracking_uri('databricks')
    mlflow.set_experiment("/Users/timkakhanovich@gmail.com/Word2Vec/v1")

    with mlflow.start_run(run_name="Word2Vec model v1.0"):
        mlflow.set_tags({
            'Python': '.'.join(map(str, sys.version_info[:3])), 
            'Device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
        })
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('lr', lr)
        _, _ = train(
            model, text_dataloader, optimizer, lr=lr, scheduler=scheduler, 
            num_epochs=num_epochs, start_epoch=start_epoch, prev_losses=history, 
            device=device, folder_for_checkpoints=CHECKPOINT_PATH
        )

if __name__ == "__main__":
    main_command_line_args()
