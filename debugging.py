"""
#from ctlearn.ctlearn.tools import load_model 
from ctlearn.tools import predict_model 
from ctlearn.tools.predict_model import PredictCTLearnModel
from dl1_data_handler.reader import DLDataReader 
#from ctlearn.ctlearn.train_model import train_model
#from ctapipe.core import run_tool
import tensorflow as tf
import keras
from dl1_data_handler.reader import (
    DLDataReader,
    DLImageReader,
    ProcessType,
    LST_EPOCH,
)
from ctlearn.core.loader import DLDataLoader
from ctapipe.core.traits import (ComponentName)
from pathlib import Path
""""""
model_path = "/data3/users/dafne/model_compression/model_for_testing/ctlearn_model/"
model = keras.saving.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_model = converter.convert()
output_path = Path("/data3/users/dafne/model_compression/model_for_testing/ctlearn_quantized_model/quant_type_model.tflite" )        
output_path.write_bytes(quant_model)
"""  
"""
data_path = Path("~/data4/datasets/mrkrabs/TestDataMC/gamma_theta_10.0_az_102.199_runs1-500.dl1.h5")
model_path = "/data3/users/dafne/model_compression/model_for_testing/ctlearn_model/"
pred_path = "/data3/users/dafne/model_compression/model_for_testing/ctlearn_quantized_model/type_prediction"
PredictCTLearnModel.batch_size = 1
PredictCTLearnModel.dl1dh_reader_type = DLImageReader
PredictCTLearnModel.input_url = data_path
PredictCTLearnModel.load_type_model_from = model_path
#--PredictCTLearnModel.log_file=<Path>
PredictCTLearnModel.output_path = pred_path
PredictCTLearnModel.overwrite_tables=True
PredictCTLearnModel.dl1dh_reader_type.channels = ("cleaned_image", "cleaned_relative_peak_time")
#--DLImageReader.channels=cleaned_relative_peak_time
#--DLImageReader.image_mapper_type=BilinearMapper 
#--use-HDF5Merger
#--no-dl1-images
#--no-dl1-features




output_path = Path("/data3/users/dafne/model_compression/model_for_testing/ctlearn_quantized_model/quant_type_model.tflite" )        
#predict_model()
predict_model.MonoPredictCTLearnModel(PredictCTLearnModel()._predict_with_tflite_model(output_path))
"""


"""
from ctlearn.tools.predict_model import PredictCTLearnModel
from pathlib import Path
from ctlearn.tools import predict_model
from ctapipe.core import Tool
import logging

log_path = '/data3/users/dafne/model_compression/model_for_testing/prediction/non_quantized/py_type_log.txt'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
    ]
)

log = logging.getLogger(__name__)
log.info("Logger initialized.")

#log = logging.getLogger(__name__)

# Initialize the tool
tool = PredictCTLearnModel()

# Set parameters manually (like you'd do via a config file or CLI)
#tool.model_path = "/data3/users/dafne/model_compression/model_for_testing/ctlearn_quantized_model/quant_type_model.tflite"
#tool.output_path = "/data3/users/dafne/model_compression/model_for_testing/prediction/quantized/"
tool.model_path = "/data3/users/dafne/model_compression/model_for_testing/ctlearn_model/"
tool.output_path = "/data3/users/dafne/model_compression/model_for_testing/prediction/non_quantized/type.dl2.h5"
tool.input_url = "/data4/datasets/mrkrabs/TestDataMC/gamma_theta_10.0_az_102.199_runs1-500.dl1.h5"
tool.log_file= log_path
tool.model_type = "mono_tool"
tool.overwrite_tables = True

tool.use_HDF5Merger = False
tool.batch_size = 1
tool.dl1dh_reader_type = "DLImageReader"
tool.channels = ["cleaned_image", "cleaned_relative_peak_time"]

# Setup + run prediction
#tool.setup()
#tool.start()
#tool.run()
#tool.run()
tool.initialize()  # instead of .setup()
log.info("Tool initialized")
tool.start() 
log.info("Tool started")
#PredictCTLearnModel(Tool)
tool.setup()
log.info("Tool set-up")
tool.finish()
log.info("Tool finished")
predict_model.mono_tool()
print(" Prediction complete")

"""