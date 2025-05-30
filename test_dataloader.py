from ctlearn.core.data_loader.loader import DLDataLoader
from dl1_data_handler.reader import DLImageReader
from dl1_data_handler.reader import DLDataReader
import numpy as np
import matplotlib.pyplot as plt
from ctlearn.tools.train.pytorch.utils import read_configuration
def on_key(event):
    # Check if the "Esc" key was pressed
    if event.key == 'escape':
        plt.close(event.canvas.figure)
        exit()


config_file = "./ctlearn/tools/train/pytorch/config/training_config_iaa_neutron_training.yml"

parameters = read_configuration(config_file)

#dl1_gamma_file = ["/storage/ctlearn_data/h5_files/mc/gamma_theta_16.087_az_108.090_runs123-182.r1.dl1.h5"]
#dl1_gamma_file = ["/storage/ctlearn_data/h5_files/lstchain_real_data/DL1/dl1_LST-1.Run02929.0013.h5"]
dl1_gamma_file = ["/storage/ctlearn_data/h5_files/mc/proton_theta_16.087_az_108.090_runs1-416.r1.dl1.h5"]


dl1dh_reader = DLDataReader.from_name(
    "DLImageReader",
    input_url_signal=dl1_gamma_file,
    channels = ["cleaned_image","cleaned_peak_time"],
    #channels = ["image","peak_time"],
    # input_url_background=sorted(self.input_url_background),
    # parent=self,
)

# dl1_reader = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)
# dataloader = DLDataLoader.create("pytorch")

random_seed = 0
indices = list(range(dl1dh_reader._get_n_events()))
training_loader = DLDataLoader.create(
    framework="pytorch",
    DLDataReader=dl1dh_reader,
    indices=indices,    
    tasks=["type","energy","skydirection","cameradirection","hillas"],
    batch_size=32,
    random_seed=0,
    sort_by_intensity=False,
    stack_telescope_images=False,
    use_augmentation=False,
    parameters = parameters
)

for batch_idx, (features, labels) in enumerate(training_loader):

    plt.rcParams['keymap.quit'].append(' ')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes.ravel()
    # fig.canvas.mpl_connect('key_press_event', lambda evt: print(repr(evt.key)))
    fig.canvas.mpl_connect('key_press_event', on_key)

    if len(features)>0:
        for id in range(len(features["image"])):
            # gammaness = 0
            # gammaness = labels["gammaness"][id]

            # if gammaness>0.9:
                     
            image = features["image"][id]
            # clean_image = features["clean_image"]
            peak_time = features["peak_time"][id]
            # clean_peak_time = features["clean_peak_time"]
            labels_class = labels["type"][id]

            hillas_intensity = features["hillas"]["hillas_intensity"][id]
            leakage_pixels_width_1= features["hillas"]["leakage_pixels_width_1"][id] 
            leakage_pixels_width_2= features["hillas"]["leakage_pixels_width_2"][id]

            leakage_intensity_width_1 = features["hillas"]["leakage_intensity_width_1"][id]
            leakage_intensity_width_2 = features["hillas"]["leakage_intensity_width_2"][id]

            morphology_n_islands= features["hillas"]["morphology_n_islands"][id]
            image = np.transpose(image, (1, 2, 0))
            # clean_image = np.transpose(clean_image, (1, 2, 0))

            peak_time = np.transpose(peak_time, (1, 2, 0))
            # clean_peak_time = np.transpose(clean_peak_time, (1, 2, 0))

            ax[0].set_title(
                f"Charge:\n Hillas Intensity:{hillas_intensity} \n Leakage_p_w_2: {leakage_pixels_width_2} \n Leakage_i_w_2: {leakage_intensity_width_2} \n morphology_n_islands: {morphology_n_islands} \n labels_class: {str(labels_class)}" , fontsize=8
            )
            # ax[1].set_title(
            #     f"Peak time: \n Gammaness: {gammaness}" , fontsize=8
            # )    
            ax[0].imshow(image, cmap="viridis")
            ax[1].imshow(peak_time, cmap="viridis")
        
            # print(f"Gammaness: {gammaness}")
            # if leakage_intensity_width_2<0.2:
            # print(f"found {leakage_intensity_width_2}")
            plt.show()
            # plt.show(block=False)
            # plt.pause(1)
            # print(batch_idx)
        plt.close()

    ii = 0