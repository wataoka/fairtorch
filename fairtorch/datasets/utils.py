from torch.utils.data.dataset import Subset

def devide(dataset, test_rate=0.2):
    total_size = len(dataset)
    train_size = int(total_size*(1-test_rate))

    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, total_size))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset