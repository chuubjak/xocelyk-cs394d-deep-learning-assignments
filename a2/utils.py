from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        self.dataset_path = dataset_path
        self.data = []
        data = csv.reader(open(dataset_path + '/labels.csv', 'r'))
        first = True
        for row in data:
            if first:
                first = False
                continue
            self.data.append(row)
        self.label_to_idx = {label: idx for idx, label in enumerate(LABEL_NAMES)}

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        filename = self.data[idx][0]
        label = self.data[idx][1]
        img = Image.open(self.dataset_path + '/' + filename)
        img = transforms.ToTensor()(img)
        return (img, self.label_to_idx[label])


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
