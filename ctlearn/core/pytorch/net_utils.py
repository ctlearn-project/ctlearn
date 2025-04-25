import importlib
import torch
import numpy as np
import os.path
import pickle
from matplotlib import pyplot as plt
from skimage.filters import gabor_kernel
from skimage.transform import resize
import onnx
from onnxsim import simplify
import warnings
from ctlearn.core.ctlearn_enum import Task, Mode 

#-------------------------------------------------------------------------------------------------------------------
def create_model(model_parameters):

    try:
        module_name = "ctlearn.core.pytorch.nets.models"
        model_type = model_parameters["model_name"]
        model_params = model_parameters["parameters"]

        # Construct full class path (you should provide the full path including the module)
        full_class_path = f"ctlearn.core.pytorch.nets.models.{model_type}"

        # Resolve the class
        module = importlib.import_module(full_class_path)
        module = getattr(module, model_type)
        model_class = getattr(module, model_type)
        # Now, instantiate the model with the parameters
        model_net = model_class(**model_params)
        return model_net

    except AttributeError:
        raise ValueError(f"Model class {model_type} not found in module {module_name}.")
    except TypeError as e:
        raise ValueError(f"Error instantiating model {model_type}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")
#-------------------------------------------------------------------------------------------------------------------

class ModelHelper:
    # -------------------------------------------------------------------------------------------------------------
    def GetNumParamters(self):

        numel_list = [
            p.numel() for p in self.model.parameters() if p.requires_grad == True
        ]
        return sum(numel_list), numel_list
    # -------------------------------------------------------------------------------------------------------------
    def savePickle(path, fileName, data):

        saveFile = os.path.join(path, fileName)
        File = open(saveFile, "ab")
        pickle.dump(data, File)
    # -------------------------------------------------------------------------------------------------------------
    def loadPickle(path, fileName):
        loadFile = os.path.join(path, fileName)
        File = open(loadFile, "rb")
        data = pickle.load(File)

        return data
    # -------------------------------------------------------------------------------------------------------------
    def plotImage(img, permute=True):

        if img.is_leaf == False:
            img = img.detach()

        if permute and len(img.shape) == 3:
            img = img.permute(1, 2, 0)

        plt.imshow(img, cmap="gray")
        plt.show()
    # -------------------------------------------------------------------------------------------------------------
    def GaborKernels(size=7, showPlots=False):

        # prepare filter bank kernels
        kernels = []
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):  # range(8):
            # theta = theta / 4. * np.pi
            # for sigma in (3):
            sigma = 3
            #        for frequency in (0.05, 0.25):
            for frequency in (0.15, 0.25, 0.35, 0.45, 0.55):
                kernel = np.real(
                    gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                )

                kernel = resize(kernel, [size, size])
                kernels.append(kernel)

                if showPlots:
                    print(
                        "Theta: ",
                        theta,
                        " Sigma: ",
                        sigma,
                        " Frequency: ",
                        frequency,
                        " Kernel size:",
                        kernel.shape,
                    )
                    plt.imshow(kernel)
                    plt.show()

        return kernels
    # -------------------------------------------------------------------------------------------------------------
    def saveModel(model, data_path, filename):
        print("Saving model: ", filename)

        torch.save(model.state_dict(), os.path.join(data_path, filename))
    # -------------------------------------------------------------------------------------------------------------
    def loadModel(model, data_path, filename, mode, device_str='cpu'):

        if os.path.isfile(os.path.join(data_path, filename)):
            print("Loading model: ", filename)
            
            # TODO: Test weights_only=True. It is getting this warning:
            # "FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature." 
            pretrained_dict = torch.load(os.path.join(data_path, filename),map_location=torch.device(device_str),weights_only=False)

            if type(pretrained_dict) == dict and "state_dict" in pretrained_dict :
                pretrained_dict = pretrained_dict["state_dict"]

            if type(pretrained_dict) == dict and "model_state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["model_state_dict"]


            model_dict = model.state_dict()

            # Remove the prefix pattern from the state dict.
            modified_dict = {}
            prefix = "model.0."
            for key in pretrained_dict:
                # Check if the key start with 'model.'
                if key.startswith(prefix):
                    # Remove the pattern 'model.' and save the value with the new key
                    new_key = key.replace(prefix, "")
                    modified_dict[new_key] = pretrained_dict[key]

            if len(modified_dict) > 0:
                pretrained_dict = modified_dict

            # 1. filter out unnecessary keys
            # pretrained_dict = {
            #     k: v for k, v in pretrained_dict.items() if k in model_dict 
            # }
            # Filter out keys that do not match in name or dimensions
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()
            }            

            if (len(model_dict)!=len(pretrained_dict) or set(model_dict.keys()) != set(pretrained_dict.keys())):

                pretrain_len = len(pretrained_dict)
                model_len = len(model_dict)
                unique_pretrained = set(pretrained_dict.keys()) - set(model_dict.keys())
                unique_model = set(model_dict.keys()) - set(pretrained_dict.keys())

                if (mode!=Mode.train and mode!=Mode.tunning):
                    raise ValueError(f"Error Loading the model. Pretrained Dict lenght: {pretrain_len} Model Dict lenght: {model_len}. Differences -> Pretrained keys: {unique_pretrained}, Model keys: {unique_model}")
                else:
                    warnings.warn(
                        f"Warning Loading the model. Pretrained Dict length: {pretrain_len} Model Dict length: {model_len}. "
                        f"Differences -> Pretrained keys: {unique_pretrained}, Model keys: {unique_model}",
                        UserWarning
                    )
                    
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict, strict=False)

            # use_cuda = torch.cuda.is_available()
            # device = torch.device(device_str if use_cuda else "cpu")
            device = torch.device(device_str)
            model.to(device)

            # model.load_state_dict(torch.load(data_path + filename), strict=False)
            print("Model Loaded.")
        else:
            print(f"CheckPoint file does not exist: {filename}")
            if mode != Mode.train:
                exit()

        return model
    # -------------------------------------------------------------------------------------------------------------
    def exportOnnx(model, dummy_input, onnx_name, input_names, output_names):

        torch.onnx.export(
            model,
            dummy_input,
            onnx_name + ".onnx",
            verbose=True,
            input_names=input_names,
            output_names=output_names,
        )

        # pip3 install -U pip && pip3 install onnxsim
        # load your predefined ONNX model
        model = onnx.load(onnx_name + ".onnx")

        # convert model
        model_simp, check = simplify(model)

        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_name + "_simp.onnx")
    # -------------------------------------------------------------------------------------------------------------
