import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(THIS_FILE_DIR, '..')
GERMAN_SEX_PATH = os.path.join(BASE_DIR, 'data', 'german', 'sex.npy')

class German(Dataset):

    def __init__(self, data_numeric_path, transform=None, protected_attribute_names=['sex'], sensitive_attribute_path=GERMAN_SEX_PATH, preprocess='min-max'):
        self.transform = transform
        self.data_names = ['status_of_credit', 'duration_in_month', 'credit_history',
                         'credit_amout', 'savings', 'job', 'feature7', 'housing',
                         'property', 'age', 'plans', 'feature12', 'feature13',
                         'feature14', 'feature15', 'feature16', 'feature17',
                         'feature18', 'feature19', 'feature20', 'feature21',
                         'feature22', 'feature23', 'feature24']
        self.label_names = ['label']
        self.df = pd.read_csv(data_numeric_path, delimiter='\s+', names=(self.data_names + self.label_names))
        self.data = self.df[self.data_names].values
        self.label = self.df[self.label_names].values
        self.sensitive_attribute = np.array(np.load(sensitive_attribute_path))
        if preprocess=='min-max':
            self.data = self.min_max(self.data, axis=1)
            self.label = self.min_max(self.label, axis=0)
            self.sensitive_attribute = self.min_max(self.sensitive_attribute, axis=0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        out_sensitive_attribute = self.sensitive_attribute[idx]

        return out_data, out_label, out_sensitive_attribute
    
    def min_max(self, x, axis=None):
        x_min = x.min(axis=axis, keepdims=True)
        x_max = x.max(axis=axis, keepdims=True)
        return (x-x_min)/(x_max-x_min)