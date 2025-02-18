import torch
from src.metrics_and_losses import (per_prop_mse, per_prop_stable_mse, per_prop_r2)
import os
import numpy as np

class Checkpointing:
    def __init__(self, 
                 property_names, 
                 model, 
                 artefact_dir, 
                 model_type, 
                 smoothing_alpha):
        
        self.counter = [0] * len(property_names) 
        self.best_losses = None
        self.property_names = property_names
        self.model = model.model
        self.artefact_dir = artefact_dir
        self.model_type = model_type
        self.checkpoints_dir = os.path.join(self.artefact_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.smoothed_losses = None
        self.smoothing_alpha = smoothing_alpha


    
    def __call__(self, block_pred, block_target, total_loss):
  
        stop_training = False
        
        losses = self.loss_calculator(block_pred, block_target)
        
        if self.model_type == 'mtl-glc':
            losses = [total_loss]*len(losses)
            print(f"Average loss: {losses[0]:.6f}")

        if self.smoothed_losses is None:
            self.smoothed_losses = losses[:] 
            self.best_losses = losses[:]
            for i in range(len(losses)):
                self.save_checkpoint(i)
        else:
            for i in range(len(losses)):
                self.smoothed_losses[i] = (
                    self.smoothing_alpha * losses[i] + 
                    (1 - self.smoothing_alpha) * self.smoothed_losses[i])

        for i in range(len(losses)):
            if self.smoothed_losses[i] > self.best_losses[i]:
                self.counter[i] += 1
                print(f"{self.property_names[i]}: current loss =  {self.smoothed_losses[i]:.6f} > best loss = {self.best_losses[i]:.6f}")
            else:
                self.save_checkpoint(i)
                self.best_losses[i] = self.smoothed_losses[i]
                self.counter[i] = 0 
                print(f"Validation loss decreased for {self.property_names[i]}, "
                      f"checkpoint saved, current loss = {self.smoothed_losses[i]:.6f}")

        if all(not param.requires_grad for param in self.model.parameters()):
            stop_training = True

        return stop_training

    def loss_calculator(self, block_pred, block_target):
        mse_list = per_prop_mse(block_pred, block_target)
        return mse_list

    def save_checkpoint(self, i):
   
        i_prop = i
        if self.model_type in ['acs', 'mtl-glc']:
            i += 1
            torch.save(
                self.model[0].state_dict(), 
                os.path.join(self.checkpoints_dir, f"{self.property_names[i_prop]}_backbone.pth"))
        torch.save(
            self.model[i].state_dict(), 
            os.path.join(self.checkpoints_dir, f"{self.property_names[i_prop]}.pth"))

    def reload_checkpoint(self, i):
 
        i_prop = i
        if self.model_type in ['acs', 'mtl-glc']:
            i += 1
        self.model[i].load_state_dict(
            torch.load(
                os.path.join(self.checkpoints_dir, f"{self.property_names[i_prop]}.pth")))
        print(f"Reloaded best checkpoint for {self.property_names[i_prop]}")

    def reload_backbone(self, i):
 
        self.model[0].load_state_dict(
            torch.load(
                os.path.join(self.checkpoints_dir, f"{self.property_names[i]}_backbone.pth")))
