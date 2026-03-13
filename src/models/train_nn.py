import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from src.models.pytorch_mlp import FraudMLP, FocalLoss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

def train_nn(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
             epochs: int = 50, batch_size: int = 1024, lr: float = 1e-3, device: str = 'cpu'):
    """
    Trains PyTorch MLP with Early Stopping.
    """
    input_dim = X_train.shape[1]
    model = FraudMLP(input_dim=input_dim).to(device)
    
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Handle single dimension arrays and pandas Series vs numpy array
    y_t = y_train.values if hasattr(y_train, 'values') else y_train
    y_v = y_val.values if hasattr(y_val, 'values') else y_val

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_t))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_v))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).view(-1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                preds = torch.sigmoid(outputs)
                all_targets.append(batch_y.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Load best weights
    model.load_state_dict(early_stopping.best_model_weights)
    return model
