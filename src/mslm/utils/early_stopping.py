import numpy as np

class EarlyStopping:
    def __init__(self, patience:int=10, threshold:float=0.002, verbose:bool=False):
        self.best_loss = float("inf")
        self.patience = patience
        self._patience = patience
        self.verbose = verbose
        self.stop = False
        self.treshold = threshold

    def __call__(self, val_loss):
        if np.isnan(val_loss):
            if self.verbose: print("Giorgio desgraciado *Christian desgraciado \nIgnorando epoch")
            return
        
        if val_loss < self.best_loss - self.treshold:
            self.best_loss = val_loss
            self.patience = self._patience
        else:
            self.patience -= 1
        
        if self.patience == 0:
            if self.verbose: print("Early Stopping")
            self.stop = True