from copy import deepcopy

class EarlyStopping:
    def __init__(self, patience = 10, delta = 0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.epoch = -1
    
    def __call__(self, val_loss, model):
        score = -val_loss
        self.epoch += 1
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_epoch = self.epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_epoch = self.epoch
            self.counter = 0
            
    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_epoch = None
        
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)