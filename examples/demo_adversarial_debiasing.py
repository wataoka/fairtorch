import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')
from fairtorch.datasets import German
from fairtorch.datasets.utils import devide
from fairtorch.algo.inprocess import Classifier, Adversary



# path
HOME_DIR: str = os.path.expanduser('~')
COMPAS_DATA_PATH: str = os.path.join(HOME_DIR, 'datasets', 'COMPAS', 'compas-scores-two-years.csv')
GERMAN_DATA_PATH: str = os.path.join(HOME_DIR, 'datasets', 'german', 'german.data')
GERMAN_DATA_NUMERIC_PATH: str = os.path.join(HOME_DIR, 'datasets', 'german', 'german.data-numeric')

if __name__ == "__main__":

    # hyper parameters
    n_epochs = 50
    n_features = 24
    n_sensitive = 1
    batch_size = 128

    # dataset
    dataset = German(data_numeric_path=GERMAN_DATA_NUMERIC_PATH)
    train_dataset, test_dataset = devide(dataset, test_rate=0.2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # generate classifier
    clf_model = Classifier(n_features=n_features)
    clf_criterion = nn.BCELoss()
    clf_optimizer = optim.Adam(clf_model.parameters())

    # generate adversary
    adv_model = Adversary(n_sensitive=n_sensitive)
    adv_criterion = nn.BCELoss(reduction='none')
    adv_optimizer = optim.Adam(adv_model.parameters())

    # train
    for epoch in range(n_epochs):

        clf_loss_list = []
        adv_loss_list = []

        # train adversary on batches
        for x, y, s in train_loader:
            adv_model.zero_grad()
            pred_y = clf_model(x)
            pred_s = adv_model(pred_y)
            adv_loss = adv_criterion(pred_s, s).mean()
            adv_loss.backward()
            adv_optimizer.step()
        
        adv_loss_list.append(adv_loss.item())

        # train classifier on single batch
        for x, y, s in train_loader:
            pass

        clf_model.zero_grad()
        pred_y = clf_model(x)
        pred_s = adv_model(pred_y)
        clf_loss = clf_criterion(pred_y, y) - adv_criterion(pred_s, s).mean()
        clf_loss.backward()
        clf_optimizer.step()

        clf_loss_list.append(clf_loss.item())

        # show result
        ave_clf_loss = sum(clf_loss_list)/len(clf_loss_list)
        ave_adv_loss = sum(adv_loss_list)/len(adv_loss_list)
        print('Epoch: {}/{}\tclf loss: {:.3f}\tadv loss: {:.3f}'.format(epoch+1, n_epochs, ave_clf_loss, ave_adv_loss))
