from ctlearn.core.data_loader.loader import DLDataLoader
from dl1_data_handler.reader import DLImageReader
from dl1_data_handler.reader import DLDataReader
import numpy as np
import matplotlib.pyplot as plt
from ctlearn.tools.train.pytorch.utils import read_configuration
import os
def on_key(event):
    # Check if the "Esc" key was pressed
    if event.key == 'escape':
        plt.close(event.canvas.figure)
        exit()


 
gamma_dir = "/storage/ctlearn_data/h5_files/mc/gamma-diffuse/"

gamma_list= [
"gamma_theta_16.087_az_108.090_runs123-182.dl1.h5", 
# "gamma_theta_16.087_az_108.090_runs1-62.dl1.h5", 
# "gamma_theta_16.087_az_108.090_runs183-242.dl1.h5", 

"gamma_theta_16.087_az_251.910_runs121-180.dl1.h5", 
# "gamma_theta_16.087_az_251.910_runs1-60.dl1.h5", 
# "gamma_theta_16.087_az_251.910_runs181-240.dl1.h5", 

"gamma_theta_23.161_az_260.739_runs129-187.dl1.h5", 
# "gamma_theta_23.161_az_260.739_runs188-246.dl1.h5", 
# "gamma_theta_23.161_az_260.739_runs247-305.dl1.h5", 

"gamma_theta_23.161_az_99.261_runs118-176.dl1.h5", 
# "gamma_theta_23.161_az_99.261_runs1-59.dl1.h5", 
# "gamma_theta_23.161_az_99.261_runs177-235.dl1.h5", 

"gamma_theta_30.390_az_266.360_runs121-180.dl1.h5", 
# "gamma_theta_30.390_az_266.360_runs1-60.dl1.h5", 
# "gamma_theta_30.390_az_266.360_runs181-240.dl1.h5", 

"gamma_theta_30.390_az_93.640_runs121-180.dl1.h5", 
"gamma_theta_30.390_az_93.640_runs1-60.dl1.h5", 
# "gamma_theta_30.390_az_93.640_runs181-240.dl1.h5", 

"gamma_theta_37.661_az_270.641_runs121-180.dl1.h5", 
# "gamma_theta_37.661_az_270.641_runs1-60.dl1.h5", 
# "gamma_theta_37.661_az_270.641_runs181-240.dl1.h5", 
# "gamma_theta_37.661_az_270.641_runs241-300.dl1.h5",

"gamma_theta_37.661_az_89.359_runs121-180.dl1.h5", 
# "gamma_theta_37.661_az_89.359_runs1-60.dl1.h5", 
# "gamma_theta_37.661_az_89.359_runs181-240.dl1.h5", 
# "gamma_theta_37.661_az_89.359_runs241-300.dl1.h5",

"gamma_theta_6.000_az_180.000_runs121-180.dl1.h5", 
# "gamma_theta_6.000_az_180.000_runs1-60.dl1.h5", 
# "gamma_theta_6.000_az_180.000_runs181-240.dl1.h5", 
# "gamma_theta_6.000_az_180.000_runs241-300.dl1.h5",

"gamma_theta_9.579_az_126.888_runs121-180.dl1.h5", 
# "gamma_theta_9.579_az_126.888_runs1-60.dl1.h5", 
# "gamma_theta_9.579_az_126.888_runs181-240.dl1.h5", 
# "gamma_theta_9.579_az_126.888_runs241-300.dl1.h5",

"gamma_theta_9.579_az_233.112_runs121-180.dl1.h5", 
# "gamma_theta_9.579_az_233.112_runs1-60.dl1.h5", 
# "gamma_theta_9.579_az_233.112_runs181-240.dl1.h5",
# "gamma_theta_9.579_az_233.112_runs241-300.dl1.h5",
]


proton_dir = "/storage/ctlearn_data/h5_files/mc/protons/"

proton_list=[
"proton_theta_16.087_az_108.090_runs1-416.dl1.h5",
# "proton_theta_16.087_az_108.090_runs417-834.dl1.h5",
# "proton_theta_16.087_az_108.090_runs835-1250.dl1.h5",
"proton_theta_16.087_az_251.910_runs1-417.dl1.h5",
# "proton_theta_16.087_az_251.910_runs418-834.dl1.h5",
# "proton_theta_16.087_az_251.910_runs835-1250.dl1.h5",
"proton_theta_23.161_az_260.739_runs1-417.dl1.h5",
# "proton_theta_23.161_az_260.739_runs418-834.dl1.h5",
# "proton_theta_23.161_az_260.739_runs835-1250.dl1.h5",
"proton_theta_23.161_az_99.261_runs1-417.dl1.h5",
# "proton_theta_23.161_az_99.261_runs418-832.dl1.h5",
# "proton_theta_23.161_az_99.261_runs833-1250.dl1.h5",
"proton_theta_30.390_az_266.360_runs1-416.dl1.h5",
# "proton_theta_30.390_az_266.360_runs417-834.dl1.h5",
# "proton_theta_30.390_az_266.360_runs835-1250.dl1.h5",
"proton_theta_30.390_az_93.640_runs1-420.dl1.h5",
# "proton_theta_30.390_az_93.640_runs421-834.dl1.h5",
# "proton_theta_30.390_az_93.640_runs835-1250.dl1.h5",
"proton_theta_37.661_az_270.641_runs1-421.dl1.h5",
# "proton_theta_37.661_az_270.641_runs422-836.dl1.h5",
# "proton_theta_37.661_az_270.641_runs837-1250.dl1.h5",
"proton_theta_37.661_az_89.359_runs1-406.dl1.h5",
# "proton_theta_37.661_az_89.359_runs408-867.dl1.h5",
# "proton_theta_37.661_az_89.359_runs868-1250.dl1.h5",
"proton_theta_6.000_az_180.000_runs1-416.dl1.h5",
# "proton_theta_6.000_az_180.000_runs417-830.dl1.h5",
# "proton_theta_6.000_az_180.000_runs831-1250.dl1.h5",
"proton_theta_9.579_az_126.888_runs1-417.dl1.h5",
# "proton_theta_9.579_az_126.888_runs418-834.dl1.h5",
# "proton_theta_9.579_az_126.888_runs835-1250.dl1.h5",

"proton_theta_9.579_az_233.112_runs1-417.dl1.h5",
# "proton_theta_9.579_az_233.112_runs418-834.dl1.h5",
# "proton_theta_9.579_az_233.112_runs835-1250.dl1.h5",
]


config_file = "./ctlearn/tools/train/pytorch/config/training_config_iaa_neutron_training.yml"


# dl1dh_reader = DLDataReader.from_name(
#     "DLImageReader",
#     input_url_signal=[file],
#     channels = ["cleaned_image","cleaned_peak_time"],
#     # input_url_background=sorted(self.input_url_background),
#     # parent=self,
# )

cnt_type=0

for file_name in gamma_list:
    dl1dh_reader = DLDataReader.from_name(
        "DLImageReader",
        input_url_signal=[os.path.join(gamma_dir,file_name)],
        channels = ["cleaned_image","cleaned_peak_time"],
        # input_url_background=sorted(self.input_url_background),
        # parent=self,
    )
    print(f"file: {file_name} events: {dl1dh_reader.n_signal_events}")
    cnt_type+=dl1dh_reader.n_signal_events


# for file_name in proton_list:
#     dl1dh_reader = DLDataReader.from_name(
#         "DLImageReader",
#         input_url_signal=[os.path.join(proton_dir,file_name)],
#         channels = ["cleaned_image","cleaned_peak_time"],
#         # input_url_background=sorted(self.input_url_background),
#         # parent=self,
#     )
#     print(f"file: {file_name} events: {dl1dh_reader.n_signal_events}")
#     cnt_type+=dl1dh_reader.n_signal_events

print("cnt_type: ", cnt_type)


gamma_cnt = "3389398"
proton_cnt = "3590228"


gamma_cnt =  1234488
proton_cnt = 1249970

# # dl1_reader = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)
# # dataloader = DLDataLoader.create("pytorch")
# parameters = read_configuration(config_file)

# random_seed = 0
# indices = list(range(dl1dh_reader._get_n_events()))
# training_loader = DLDataLoader.create(
#     framework="pytorch",
#     DLDataReader=dl1dh_reader,
#     indices=indices,     
#     tasks=["type","energy","skydirection","cameradirection","hillas"],
#     batch_size=32,
#     random_seed=0,
#     sort_by_intensity=False,
#     stack_telescope_images=False,
#     parameters= parameters,
#     use_augmentation=True
# )


# print(len(training_loader))
# ii=0
# for batch_idx, (features, labels) in enumerate(training_loader):

#     plt.rcParams['keymap.quit'].append(' ')
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#     ax = axes.ravel()
#     # fig.canvas.mpl_connect('key_press_event', lambda evt: print(repr(evt.key)))
#     fig.canvas.mpl_connect('key_press_event', on_key)

#     if len(features)>0:
#         for id in range(len(features["image"])):
#             # gammaness = 0
#             # gammaness = labels["gammaness"][id]

#             # if gammaness>0.9:
                     
#             image = features["image"][id]
#             # clean_image = features["clean_image"]
#             peak_time = features["peak_time"][id]
#             # clean_peak_time = features["clean_peak_time"]
#             labels_class = labels["type"][id]

#             hillas_intensity = features["hillas"]["hillas_intensity"][id]
#             leakage_pixels_width_1= features["hillas"]["leakage_pixels_width_1"][id] 
#             leakage_pixels_width_2= features["hillas"]["leakage_pixels_width_2"][id]

#             leakage_intensity_width_1 = features["hillas"]["leakage_intensity_width_1"][id]
#             leakage_intensity_width_2 = features["hillas"]["leakage_intensity_width_2"][id]

#             morphology_n_islands= features["hillas"]["morphology_n_islands"][id]
#             image = np.transpose(image, (1, 2, 0))
#             # clean_image = np.transpose(clean_image, (1, 2, 0))

#             peak_time = np.transpose(peak_time, (1, 2, 0))
#             # clean_peak_time = np.transpose(clean_peak_time, (1, 2, 0))

#             ax[0].set_title(
#                 f"Charge:\n Hillas Intensity:{hillas_intensity} \n Leakage_p_w_2: {leakage_pixels_width_2} \n Leakage_i_w_2: {leakage_intensity_width_2} \n morphology_n_islands: {morphology_n_islands} \n labels_class: {str(labels_class)}" , fontsize=8
#             )
#             # ax[1].set_title(
#             #     f"Peak time: \n Gammaness: {gammaness}" , fontsize=8
#             # )    
#             ax[0].imshow(image, cmap="viridis")
#             ax[1].imshow(peak_time, cmap="viridis")
        
#             # print(f"Gammaness: {gammaness}")
#             # if leakage_intensity_width_2<0.2:
#             # print(f"found {leakage_intensity_width_2}")
#             plt.show()
#             # plt.show(block=False)
#             # plt.pause(1)
#             # print(batch_idx)
#         plt.close()

#     ii = 0