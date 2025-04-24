# from .keras_loader import KerasDLDataLoader
# from .pytorch_loader import PyTorchDLDataLoader

class DLDataLoader:
    @staticmethod
    def create(framework, **kwargs):

        dataloader = None 
        if framework == "keras":
            try:
                from .keras_loader import KerasDLDataLoader
                dataloader = KerasDLDataLoader(**kwargs)
            except ImportError as e:
                raise ImportError(f"Not possible to import KerasDLDataLoader: {e}") from e
             
        elif framework == "pytorch":
            try:
                from .pytorch_loader import PyTorchDLDataLoader
                dataloader = PyTorchDLDataLoader(**kwargs)
            except ImportError as e:
                raise ImportError(f"Not possible to import PyTorchDLDataLoader: {e}") from e
 
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return dataloader
