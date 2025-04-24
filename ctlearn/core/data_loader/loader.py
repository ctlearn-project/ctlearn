from .keras_loader import KerasDLDataLoader
from .pytorch_loader import PyTorchDLDataLoader

class DLDataLoader:
    @staticmethod
    def create(framework, **kwargs):
        if framework == "keras":
            return KerasDLDataLoader(**kwargs)
        elif framework == "pytorch":
            return PyTorchDLDataLoader(**kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")