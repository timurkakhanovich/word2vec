import argparse
import pickle
import textwrap

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import Word2Vec

class ModelConfig():
    """
    Gets arguments from command line to initialize model.
    STRICTLY REQUIRED: 
    * for chosen checkpoint: batch_size, num_epochs, device;
    * if checkpoint is not chosen: lr, batch_size, num_epochs, device
    """

    def __init__(self, checkpoint_path='/'):
        self.checkpoint_path = checkpoint_path

    def load_from_checkpoint(self, check_epoch, batch_size, 
                            num_epochs, device):
        checkpoint = torch.load(
            self.checkpoint_path + f'checkpoint_epoch_{check_epoch}.pt', map_location=device
        )
        
        epoch = checkpoint['epoch']
        model = checkpoint['model_architecture'].to(device)
        model.device = device
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = checkpoint['lr']
        history = checkpoint['whole_history']
        
        if checkpoint['scheduler']:
            scheduler = checkpoint['scheduler']
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'model': model, 
            'optimizer': optimizer, 
            'scheduler': scheduler if checkpoint['scheduler'] else None, 
            'history': history, 
            'batch_size': batch_size, 
            'start_epoch': epoch, 
            'lr': lr, 
            'num_epochs': num_epochs, 
            'device': device
        }

    def get_arguments(self):
        parser = argparse.ArgumentParser(
            prog='PROG',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent("""
            Gets arguments from command line to initialize model.
            STRICTLY REQUIRED: 
            * for chosen checkpoint: batch_size, num_epochs, device, K;
            * if checkpoint is not chosen: lr, emb_dim (default: 300), K, batch_size, num_epochs, device.
            """)
        )
        parser.add_argument('--checkpoint', type=int, help='Loading from checkpoint')
        parser.add_argument('--K', type=int, help='K negative sampling')
        parser.add_argument('--emb_dim', type=int, help='Embedding dim for training', default=300)
        parser.add_argument('--batch_size', type=int, help='Batch size')
        parser.add_argument('--device', type=str, help='Device type')
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--num_epochs', type=int, help='Num epochs for training')
        parser.add_argument('--T_max', type=int, help='T_max for cosine annealing', default=785)

        args = parser.parse_args()

        if args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
            if args.device == 'cuda':
                print('CUDA is not available. Switched to CPU')
        
        if args.checkpoint:
            config_data = self.load_from_checkpoint(check_epoch=args.checkpoint, 
                                            batch_size=args.batch_size,
                                            num_epochs=args.num_epochs, 
                                            device=device)
            
            config_data.update({'K': args.K})

            return config_data
        else:
            with open('vocab.pkl', 'rb') as fout:
                vocab = pickle.load(fout)

            model = Word2Vec(vocab_size=len(vocab), embedding_dim=args.emb_dim).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            if args.T_max is not None: 
                scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
            else:
                scheduler = None
                
            return {
                'model': model, 
                'optimizer': optimizer, 
                'scheduler': scheduler, 
                'history': dict(), 
                'batch_size': args.batch_size, 
                'start_epoch': -1, 
                'lr': args.lr, 
                'num_epochs': args.num_epochs, 
                'device': device, 
                'K': args.K
            }