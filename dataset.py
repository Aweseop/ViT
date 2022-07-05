
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler


def load_MNIST(batch_size, n_fold: int = 5) :
    transform = T.Compose([
                    T.ToTensor()
                ])

    dataset_train = torchvision.datasets.MNIST('dataset', train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.MNIST('dataset', train=False, download=True, transform=transform)

    train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(0, len(dataset_train) * (n_fold - 1) // n_fold)))
    val = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(len(dataset_train) * (n_fold - 1) // n_fold, len(dataset_train))))
    test = DataLoader(dataset_test, batch_size=batch_size)

    return train, val, test