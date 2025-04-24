import torch
from torch.utils.data import Dataset
from .base_loader import BaseDLDataLoader

class PyTorchDLDataLoader(Dataset, BaseDLDataLoader):
    def __init__(self, DLDataReader, indices, tasks, **kwargs):
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.sort_by_intensity = kwargs.get('sort_by_intensity', False)
        self.stack_telescope_images = kwargs.get('stack_telescope_images', False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        # Replace with PyTorch-style tensor return
        return torch.tensor(...), torch.tensor(...)

    def on_epoch_end(self):
        pass  # Optional: implement if you're using a custom sampler