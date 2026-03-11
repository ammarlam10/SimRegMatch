import os, warnings, random, gc
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from math import isnan
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


class HuberLoss(nn.Module):
    """
    Custom Huber Loss implementation for older PyTorch versions.
    Huber loss is L2 for small errors (< delta) and L1 for large errors (>= delta).
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, input, target):
        abs_diff = torch.abs(input - target)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
         
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            return loss.mean()

from dataloaders import make_semi_loader
from models.resnet_proposed import resnet50
from models.efficientnet_wrapper import efficientnet_b0
from models.unet import unet, unet_small

from utils.saver import Saver
from utils.tqdm_config import get_tqdm_config


class SimRegMatchTrainer(object):
    def __init__(self, args):
        self.args = args
        
        self.saver = Saver(self.args)
        self.saver.save_experiment_config(self.args)
        self.experiment_dir = self.saver.experiment_dir
        print(self.experiment_dir)
        
        self.result_dir = os.path.join(self.experiment_dir, 'csv')
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.experiment_dir)
        self.labeled_loader, self.unlabeled_loader, self.valid_loader, self.test_loader = \
            make_semi_loader(self.args, num_workers=self.args.num_workers)
        
        # Use EfficientNet-B0 for so2sat_pop dataset, UNet for bayern_forest, ResNet50 for others
        # Note: softplus removed to allow full range prediction with normalized labels
        if self.args.dataset.lower() == 'so2sat_pop':
            self.model = efficientnet_b0(dropout=self.args.dropout, use_softplus=False).to(self.args.cuda)
            print("Using EfficientNet-B0 model for so2sat_pop dataset")
        elif self.args.dataset.lower() in ['bayern_forest', 'simreg_bayern_forest']:
            # Check if unet-small is specified, otherwise use regular unet
            if self.args.model.lower() == 'unet-small':
                self.model = unet_small(in_channels=3, out_channels=1, dropout=self.args.dropout).to(self.args.cuda)
                print("Using UNet-Small model (3 levels) for bayern_forest dataset (pixel-wise regression)")
            else:
                self.model = unet(in_channels=3, out_channels=1, dropout=self.args.dropout).to(self.args.cuda)
                print("Using UNet model (4 levels) for bayern_forest dataset (pixel-wise regression)")
        else:
            # Use pretrained weights for age estimation datasets (utkface, agedb)
            use_pretrained = self.args.dataset.lower() in ['utkface', 'agedb']
            self.model = resnet50(dropout=self.args.dropout, use_softplus=False, pretrained=use_pretrained).to(self.args.cuda)
            if use_pretrained:
                print(f"Using ResNet50 model with ImageNet pretrained weights for {self.args.dataset} dataset")
            else:
                print(f"Using ResNet50 model (from scratch) for {self.args.dataset} dataset")
        
        # Wrap model with DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        if self.args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.lr)
        elif self.args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=self.args.lr,
                                              momentum=self.args.momentum,
                                              weight_decay=self.args.weight_decay)

        if self.args.loss=='mse':
            self.criterion = nn.MSELoss().to(self.args.cuda)
            self.criterion_unlabel = nn.MSELoss(reduction='none').to(self.args.cuda)
        elif self.args.loss=='l1':
            self.criterion = nn.L1Loss().to(self.args.cuda)
            self.criterion_unlabel = nn.L1Loss(reduction='none').to(self.args.cuda)
        elif self.args.loss=='huber':
            # Huber loss: L2 for small errors, L1 for large errors
            # delta controls the transition point between L2 and L1 behavior
            delta = getattr(self.args, 'huber_delta', 1.0)
            print(f"Using Huber loss with delta={delta}")
            self.criterion = HuberLoss(delta=delta, reduction='mean').to(self.args.cuda)
            self.criterion_unlabel = HuberLoss(delta=delta, reduction='none').to(self.args.cuda)
        
        self.args.best_valid_loss = np.inf
        self.args.best_valid_epoch = 0
        self.cnt_train, self.cnt_valid = 0, 0
        
        # Load checkpoint if resuming
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(f"Loading checkpoint from {self.args.resume}")
                checkpoint = torch.load(self.args.resume, map_location=self.args.cuda)
                # Handle DataParallel state_dict (keys have 'module.' prefix)
                state_dict = checkpoint['model_state_dict']
                # If model is wrapped in DataParallel but checkpoint wasn't, add 'module.' prefix
                if isinstance(self.model, nn.DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                # If model is not wrapped but checkpoint was, remove 'module.' prefix
                elif not isinstance(self.model, nn.DataParallel) and any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.args.best_valid_loss = checkpoint.get('best_valid_loss', np.inf)
                self.args.best_valid_epoch = checkpoint.get('best_valid_epoch', 0)
                self.cnt_train = checkpoint.get('cnt_train', 0)
                self.cnt_valid = checkpoint.get('cnt_valid', 0)
                # Update start_epoch from checkpoint if available
                if 'epoch' in checkpoint:
                    self.args.start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed from epoch {checkpoint.get('epoch', self.args.start_epoch)}")
                print(f"Will continue from epoch {self.args.start_epoch}")
                print(f"Best validation loss: {self.args.best_valid_loss:.4f} at epoch {self.args.best_valid_epoch}")
            else:
                print(f"Warning: Checkpoint file {self.args.resume} not found. Starting from scratch.")

    def train(self, epoch):
        self.model.train()
        losses_t, losses_l, losses_u = 0.0, 0.0, 0.0
        iter_unlabeled = iter(self.unlabeled_loader)
        
        total_steps = len(self.labeled_loader.dataset) // self.args.batch_size
        with tqdm(**get_tqdm_config(total=total_steps,
                  leave=True, color='green')) as pbar:
            for idx, samples_labeled in enumerate(self.labeled_loader):
                # Load data
                inputs_l = samples_labeled['input'].to(self.args.cuda)
                labels_l = samples_labeled['label'].to(self.args.cuda)

                try:
                    samples_unlabeled = next(iter_unlabeled)
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    samples_unlabeled = next(iter_unlabeled)

                inputs_u = samples_unlabeled['weak'].to(self.args.cuda)
                inputs_s = samples_unlabeled['strong'].to(self.args.cuda)
                
                # Predict labeled examples 
                preds_x, vecs_x = self.model(inputs_l)

                # Predict strong-augmented examples (uncertainty estimation - no gradients needed)
                preds_w, vecs_w = [], []
                with torch.no_grad():  # Don't track gradients for uncertainty estimation
                    for _ in range(self.args.iter_u):
                        tmp_preds, tmp_vecs = self.model(inputs_u)
                        preds_w.append(tmp_preds.unsqueeze(-1))
                        vecs_w.append(tmp_vecs.unsqueeze(-1))
                
                preds_w = torch.cat(preds_w, dim=-1)
                vecs_w = torch.cat(vecs_w, dim=-1)
                
                # Predict strong-augmented examples
                preds_s, _ = self.model(inputs_s)
                
                # loss calculation for labeled examples
                loss_x = self.criterion(preds_x, labels_l)
                
                # loss calculation for unlabeled examples
                # Handle both scalar and pixel-wise predictions
                if preds_w.dim() == 5:  # Pixel-wise: (B, C, H, W, iter_u)
                    v_mean = torch.mean(preds_w, dim=4)  # (B, C, H, W)
                    vecs_w = torch.mean(vecs_w, dim=2)  # (B, feature_dim)
                else:  # Scalar: (B, 1, iter_u)
                    v_mean = torch.mean(preds_w, axis=2)  # (B, 1)
                    vecs_w = torch.mean(vecs_w, axis=2)  # (B, feature_dim)

                # similarity distribution
                vecs_w, vecs_x = F.normalize(vecs_w, dim=1), F.normalize(vecs_x, dim=1) 
                similaritys = vecs_w @ vecs_x.T
                similaritys = torch.softmax(similaritys/self.args.t, dim=1)
                
                # similarity-based pseudo-label
                # Handle both scalar (B, 1) and pixel-wise (B, 1, H, W) labels
                if labels_l.dim() == 4:  # Pixel-wise regression (B, 1, H, W)
                    # For pixel-wise, we need to reshape for matrix multiplication
                    B, C, H, W = labels_l.shape
                    labels_l_flat = labels_l.view(B, -1)  # (B, C*H*W)
                    v_similarity_flat = similaritys @ labels_l_flat  # (B_unlabeled, C*H*W)
                    v_similarity = v_similarity_flat.view(-1, C, H, W)  # (B_unlabeled, C, H, W)
                else:  # Scalar regression (B, 1)
                    v_similarity = similaritys @ labels_l
                
                # Clean up inputs_l and labels_l - no longer needed after similarity computation
                del inputs_l, labels_l
                
                # pseudo-label calibration
                v_mean = self.args.beta*v_mean + (1-self.args.beta)*v_similarity # beta*modelPL + (1-beta)*simPL
                
                # uncertainty estimation
                # Handle both scalar and pixel-wise predictions
                if preds_w.dim() == 5:  # Pixel-wise: (B, C, H, W, iter_u)
                    v_uncertainty = torch.pow(torch.std(preds_w, dim=4), 2)  # (B, C, H, W)
                    v_uncertainty = torch.sum(v_uncertainty, dim=(1, 2, 3))  # (B,)
                else:  # Scalar: (B, 1, iter_u)
                    v_uncertainty = torch.pow(torch.std(preds_w, axis=2), 2)  # (B, 1)
                    v_uncertainty = torch.sum(v_uncertainty, axis=1)  # (B,)
                
                # Clean up inputs_u and preds_w - no longer needed after uncertainty computation
                del inputs_u, preds_w
                
                # pseudo-label filtering
                mask = (v_uncertainty < self.args.threshold)
                loss_u_per_sample = self.criterion_unlabel(v_mean.detach(), preds_s)
                # Handle both scalar and pixel-wise loss
                if loss_u_per_sample.dim() == 4:  # Pixel-wise (B, C, H, W)
                    loss_u_per_sample = loss_u_per_sample.mean(dim=(1, 2, 3))  # (B,) - use mean to match labeled loss scale
                else:  # Scalar (B, 1)
                    loss_u_per_sample = loss_u_per_sample.sum(axis=1)  # (B,)
                loss_u = (loss_u_per_sample * mask).sum()/(int(mask.sum()))
                
                # Note: percentile arg is in [0,1] range (e.g., 0.95 = 95th percentile)
                # np.percentile expects q in [0,100], so multiply by 100
                self.args.threshold = np.percentile(v_uncertainty.detach().cpu().numpy(), q=float(self.args.percentile) * 100)           
                self.writer.add_scalar(
                    'Threshold',
                    self.args.threshold,
                    global_step=self.cnt_train
                )
                           
                if isnan(loss_u.item()):
                    loss_u = torch.tensor(0).to(self.args.cuda)
                
                loss = loss_x + self.args.lambda_u*loss_u

                # Clean up memory more aggressively (inputs_u, preds_w, inputs_l, labels_l already deleted earlier)
                del(inputs_s, preds_s, preds_x, vecs_x, vecs_w, similaritys, v_similarity, v_mean, v_uncertainty, mask)
                gc.collect()
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses_t += loss.item()
                losses_l += loss_x.item()
                losses_u += loss_u.item()
                
                self.writer.add_scalars(
                    'Train steps',
                    {'Total Loss': losses_t/(idx+1),
                    'Labeled Loss': losses_l/(idx+1),
                    'Unlabeled Loss': losses_u/(idx+1)},
                    global_step=self.cnt_train
                )
                self.cnt_train += 1
                
                description = "%-7s(%3d/%3d) Total: %.4f| Labeled: %.4f| Unlabeled: %.4f"%(
                    'SEMI', idx, total_steps, losses_t/(idx+1), losses_l/(idx+1), losses_u/(idx+1)
                )
                pbar.set_description(description)
                pbar.update(1)  

    def validation(self, epoch):
        self.model.eval()
        losses_t = 0.0
        
        total_steps = len(self.valid_loader.dataset) // self.args.batch_size + 1
        with torch.no_grad():  # Don't track gradients during validation
            with tqdm(**get_tqdm_config(total=total_steps,
                                        leave=True, color='blue')) as pbar:
                for idx, samples in enumerate(self.valid_loader):
                    inputs_l = samples['input'].to(self.args.cuda)
                    labels_l = samples['label'].to(self.args.cuda)
                    
                    preds, _ = self.model(inputs_l)
                    loss = self.criterion(preds, labels_l)
                                    
                    losses_t += loss.item()
                    
                    # Denormalize predictions and labels for metrics if normalization was used
                    labels_denorm = labels_l.detach().cpu()
                    preds_denorm = preds.detach().cpu()
                    if self.args.label_mean is not None and self.args.label_std is not None:
                        labels_denorm = labels_denorm * self.args.label_std + self.args.label_mean
                        preds_denorm = preds_denorm * self.args.label_std + self.args.label_mean
                    
                    # Reverse log-transform if it was applied
                    if getattr(self.args, 'log_transform', False):
                        labels_denorm = torch.expm1(labels_denorm)  # exp(x) - 1
                        preds_denorm = torch.expm1(preds_denorm)
                    
                    if idx == 0:
                        labels_total = labels_denorm
                        preds_total = preds_denorm
                    else:
                        labels_total = torch.cat((labels_total, labels_denorm), dim=0)
                        preds_total = torch.cat((preds_total, preds_denorm), dim=0)

                    r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
                    self.writer.add_scalars(
                        'Validation steps',
                        {'Loss': losses_t/(idx+1),
                         'MAE': mae,
                         'RMSE': rmse,
                         'R2': r2},
                        global_step=self.cnt_valid
                    )
                    self.cnt_valid += 1

                    # Clean up GPU memory after each batch
                    del inputs_l, labels_l, preds, loss
                    torch.cuda.empty_cache()

                    desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Valid", idx, total_steps, losses_t/(idx+1), r2, mae, rmse)
                    pbar.set_description(desc)
                    pbar.update(1)

            desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Valid", epoch, self.args.epochs, losses_t/(idx+1), r2, mae, rmse)
            pbar.set_description(desc)

            losses_t /= (idx+1)
            if self.args.best_valid_loss > losses_t:
                self.args.best_valid_loss = losses_t
                self.args.best_valid_epoch = epoch
                
                self.args.valid_r2 = str(r2)
                self.args.valid_mae = str(mae)
                self.args.valid_rmse = str(rmse)
                
                # Save best model state dict (for inference)
                # Remove 'module.' prefix if model is wrapped in DataParallel
                model_state = self.model.state_dict()
                if isinstance(self.model, nn.DataParallel):
                    model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
                torch.save(model_state,
                    os.path.join(self.experiment_dir, 'best_model.pth'))
                
                # Save full checkpoint (for resuming)
                # Reuse model_state from above (already has 'module.' prefix removed if needed)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_valid_loss': self.args.best_valid_loss,
                    'best_valid_epoch': self.args.best_valid_epoch,
                    'cnt_train': self.cnt_train,
                    'cnt_valid': self.cnt_valid,
                }
                torch.save(checkpoint,
                    os.path.join(self.experiment_dir, 'checkpoint.pth'))
                
                self.saver.save_experiment_config(self.args)

    def inference(self, epoch):
        weight = torch.load(os.path.join(os.path.join(self.experiment_dir, 'best_model.pth')), map_location=self.args.cuda)
        # Handle DataParallel state_dict (keys have 'module.' prefix)
        # If model is wrapped in DataParallel but checkpoint wasn't, add 'module.' prefix
        if isinstance(self.model, nn.DataParallel) and not any(k.startswith('module.') for k in weight.keys()):
            weight = {f'module.{k}': v for k, v in weight.items()}
        # If model is not wrapped but checkpoint was, remove 'module.' prefix
        elif not isinstance(self.model, nn.DataParallel) and any(k.startswith('module.') for k in weight.keys()):
            weight = {k.replace('module.', ''): v for k, v in weight.items()}
        self.model.load_state_dict(weight)
        
        self.model.eval()
        losses_t = 0.0
        
        total_steps = len(self.test_loader.dataset) // self.args.batch_size +1
        with torch.no_grad():  # Don't track gradients during inference
            with tqdm(**get_tqdm_config(total=total_steps,
                                        leave=True, color='red')) as pbar:
                for idx, samples in enumerate(self.test_loader):
                    inputs_l = samples['input'].to(self.args.cuda)
                    labels_l = samples['label'].to(self.args.cuda)
                    
                    preds, _ = self.model(inputs_l)
                    loss = self.criterion(preds, labels_l)
                                    
                    losses_t += loss.item()
                    
                    # Denormalize predictions and labels for metrics if normalization was used
                    labels_denorm = labels_l.detach().cpu()
                    preds_denorm = preds.detach().cpu()
                    if self.args.label_mean is not None and self.args.label_std is not None:
                        labels_denorm = labels_denorm * self.args.label_std + self.args.label_mean
                        preds_denorm = preds_denorm * self.args.label_std + self.args.label_mean
                    
                    # Reverse log-transform if it was applied
                    if getattr(self.args, 'log_transform', False):
                        labels_denorm = torch.expm1(labels_denorm)  # exp(x) - 1
                        preds_denorm = torch.expm1(preds_denorm)
                    
                    if idx == 0:
                        labels_total = labels_denorm
                        preds_total = preds_denorm
                    else:
                        labels_total = torch.cat((labels_total, labels_denorm), dim=0)
                        preds_total = torch.cat((preds_total, preds_denorm), dim=0)

                    # Clean up GPU memory after each batch
                    del inputs_l, labels_l, preds, loss
                    torch.cuda.empty_cache()

                    r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
                    desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Test", idx, total_steps, losses_t/(idx+1), r2, mae, rmse)
                    pbar.set_description(desc)
                    pbar.update(1)

            desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Test", epoch, self.args.epochs, losses_t/(idx+1), r2, mae, rmse)
            pbar.set_description(desc)

            r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
            losses_t /= (idx+1)
            
            self.args.best_test_loss = losses_t
            
            self.args.test_r2 = str(r2)
            self.args.test_mae = str(mae)
            self.args.test_rmse = str(rmse)
            
            labels_total, preds_total = labels_total.numpy(), preds_total.numpy()
            
            # For pixel-wise predictions, flatten to save as CSV
            if labels_total.ndim > 2:  # Pixel-wise (B, C, H, W)
                labels_total = labels_total.reshape(-1)  # Flatten all dimensions
                preds_total = preds_total.reshape(-1)

            labels_total, preds_total = pd.DataFrame(labels_total), pd.DataFrame(preds_total)
            df = pd.concat([labels_total, preds_total], axis=1)
            df.columns = ['Real', 'Pred']
            
            df.to_csv(os.path.join(self.result_dir, f'test_{str(epoch)}.csv'), index=False)
            
            self.saver.save_experiment_config(self.args)

    @staticmethod
    def regression_metrics(reals, preds):
        reals, preds = reals.numpy(), preds.numpy()
        
        # Flatten arrays for pixel-wise regression (handles both scalar and spatial predictions)
        # For scalar: (B, 1) -> (B,)
        # For spatial: (B, 1, H, W) -> (B*H*W,)
        reals = reals.flatten()
        preds = preds.flatten()
        
        r2, mae = r2_score(reals, preds), mean_absolute_error(reals, preds)
        rmse = np.sqrt(mean_squared_error(reals, preds))
        
        return r2, mae, rmse
