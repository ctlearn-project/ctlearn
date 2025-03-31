from ._version import __version__
import importlib.util


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None

__all__ = ["__version__", "is_package_available"]

 



# from ctlearn.tools.pytorch.train_pytorch_model import TrainPyTorchModel
# from ctlearn.tools.keras.train_keras_model import TrainKerasModel

# class FrameworkType(Enum):
#     KERAS = 1
#     PYTORCH = 2

# def get_framework(self,framework_type: FrameworkType):
#     if framework_type == FrameworkType.KERAS:
#         if not self.is_package_available("tensorflow"):
#             raise ImportError("TensorFlow is not installed. Cannot run Keras framework.")
#         else: 
#             fw = TrainKerasModel()

#     elif framework_type == FrameworkType.PYTORCH:
#         if not self.is_package_available("torch"):
#             raise ImportError("PyTorch is not installed. Cannot run PyTorch framework.")
#         else:            
#             fw = TrainPyTorchModel()
#     else:
#         raise ValueError("Unknown Framework")

#     return fw
