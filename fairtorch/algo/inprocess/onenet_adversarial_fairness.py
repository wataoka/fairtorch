import torch
import torch.nn as nn

class OnenetAdversarialFariness(nn.Module):

    def __init__(self, n_features, n_sensitive, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.task_head = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),
            nn.sigmoid(1),
        )

        self.fairness_head = nn.Sequential(
            nn.Linear(n_hidden, n_sensitive),
            nn.sigmoid(n_sensitive),
        )
    
    def forward(self, x):
        z = self.embedding(x)
        pred_y = self.task_head(z)
        pred_s = self.fairness_head(z)
        return pred_y, pred_s