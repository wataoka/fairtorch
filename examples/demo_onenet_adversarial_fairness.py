import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')
from fairtorch.datasets import German
from fairtorch.datasets.utils import devide
from fairtorch.algo.inprocess import OnenetAdversarialFariness



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

    # generate model
    model = OnenetAdversarialFariness(n_features=n_epochs, n_sensitive=n_sensitive)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters())

    # train
    for epoch in range(n_epochs):

        loss_list = []

        # train adversary on batches
        for x, y, s in train_loader:
            model.zero_grad()
            pred_y, pred_s = model(x)
            task_loss = criterion(pred_y, y).mean()
            fair_loss = criterion(pred_s, s).mean()
            import pdb; pdb.set_trace()
            task_loss.backward()
            optimizer.step()
        
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
