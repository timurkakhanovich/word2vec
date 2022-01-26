import torch

import mlflow
from tqdm import tqdm
import gc

def train(model, dataloaders, optimizer, lr, scheduler=None, 
          num_epochs=5, start_epoch=-1, prev_losses=list(), device=torch.device('cpu'),
          folder_for_checkpoints='/'):
    if len(prev_losses) > 0:
        history = prev_losses[:]
        curr_step = history[-1][0]

        for step, loss in prev_losses:
            mlflow.log_metric('train_loss', loss, step=step)
    else:
        history = list()
        curr_step = 1

    model.train()
    for epoch in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        running_loss = 0.0

        print("-" * 20)
        print(f"Epoch: {epoch}/{start_epoch + num_epochs}")
        print("-" * 20)
        print("Train: ")
        
        for batch_idx, (central, context, negatives) in enumerate(tqdm(dataloaders)):
            central = central.to(device)
            context = context.to(device)
            negatives = negatives.to(device)
            
            loss = model(central, context, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()
            
            running_loss += loss.item()

            if batch_idx % 100 == 99:
                mlflow.log_metric('train_loss', running_loss / (batch_idx + 1), curr_step)
                
                history.append((curr_step, running_loss / (batch_idx + 1)))

                print(f"\nRunning training loss: {running_loss / (batch_idx + 1)}")
            
            gc.collect()
            del central, context, negatives
            torch.cuda.empty_cache()
            curr_step += 1
        
        mean_train_loss = running_loss / len(dataloaders)

        print(f"Training loss: {mean_train_loss}")
        
        state = {
                'epoch': epoch,
                'batch_size_training': dataloaders.batch_size, 
                'model_architecture': model, 
                'model_state_dict': model.state_dict(), 
                'optimizer': optimizer, 
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler': scheduler if scheduler else None, 
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 
                'lr': lr, 
                'losses': history
            }

        torch.save(state, folder_for_checkpoints + f'checkpoint_epoch_{epoch % 3 + 1}.pt')
        # Logging 5 latest checkpoints.  
        mlflow.log_artifacts(folder_for_checkpoints)
        
    return model, history