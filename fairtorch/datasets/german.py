import os
import sys

import pandas as pd
from torch.utils.data import Dataset

class German(Dataset):

    def __init__(self, root, transform=None):
        self.features = ['status_of_credit', 'duration_in_month', 'credit_history',
                         'credit_amout', 'savings', 'job', 'feature7', 'housing',
                         'property', 'age', 'plans', 'feature12', 'feature13',
                         'feature14', 'feature15', 'feature16', 'feature17',
                         'feature18', 'feature19', 'feature20', 'feature21',
                         'feature22', 'feature23', 'feature24']
        self.labels = ['label']
        self.column_names = self.features + self.labels
        self.df = pd.read_csv(
            root,
            delimiter='\s+',
            names=(self.features + self.labels)
        )
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        labels = self.df.iat(idx, self.labels)
        features = self.df.loc(idx, self.features)