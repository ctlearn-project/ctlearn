import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
import os 
import seaborn as sns
import pandas as pd
import astropy.units as u
import ctaplot
import torch
# ----------------------------------------------------------------------------------------------------------
def plot_energy_resolution_error(val_energy_pred_list,val_energy_label_list,val_hillas_intensity_list):
    # Create a DataFrame to store the data
    data = pd.DataFrame({
        'energy_reco': list(val_energy_pred_list),
        'energy_true': list(val_energy_label_list),
        'hillas_intensity': list(val_hillas_intensity_list)
    })
    cut_off = 50
    filtered_data = data[data['hillas_intensity'] > cut_off]
    true_energy = u.Quantity(filtered_data['energy_true'], u.TeV)
    reco_energy = u.Quantity(filtered_data['energy_reco'], u.TeV)
    # .apply(lambda x: x[0])

    fig, ax = plt.subplots()
    ctaplot.plot_energy_resolution(true_energy, reco_energy, label="Energy resolution", ax=ax)
    # Skip warning 
    ax.get_xaxis().set_units(None)
    ax.get_yaxis().set_units(None)

    ctaplot.plot_energy_resolution_cta_requirement('north', ax=ax, color='black')
    ax.legend()
    ax.set_ylim(bottom=0, top=1.5) 
    # plt.show(block=True)
    # plt.close()

    return fig
# ----------------------------------------------------------------------------------------------------------
def plot_direction_resolution_error(val_alt_pred_list,val_az_pred_list, val_alt_label_list,val_az_label_list, val_energy_label_list,val_hillas_intensity_list):

    # Create a DataFrame to store the data
    data = pd.DataFrame({
        'alt_reco': list(val_alt_pred_list),
        'az_reco': list(val_az_pred_list),
        'alt_true': list(val_alt_label_list),
        'az_true': list(val_az_label_list),
        'energy_true': list(val_energy_label_list),
        'hillas_intensity': list(val_hillas_intensity_list)
    })
    cut_off = 50
    filtered_data = data[data['hillas_intensity'] > cut_off]
    true_energy = u.Quantity(filtered_data['energy_true'], u.TeV)

    alt_true = u.Quantity(filtered_data['alt_true'], u.rad)
    az_true = u.Quantity(filtered_data['az_true'], u.rad)

    alt_reco = u.Quantity(filtered_data['alt_reco'], u.rad)
    az_reco = u.Quantity(filtered_data['az_reco'], u.rad)

    fig, ax = plt.subplots()
    ax = ctaplot.plot_angular_resolution_per_energy(alt_true, alt_reco, az_true, az_reco, true_energy, label="Ang res (mono)")
    ctaplot.plot_angular_resolution_cta_requirement('north', ax=ax, color='black')
    ax.legend()
    ax.set_ylim(bottom=0, top=1.5) 
    return fig    
# ----------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, cm_file_name, save_folder):
    """
    Plots and saves the confusion matrix.
    
    Args:
        cm (torch.Tensor or np.ndarray): The confusion matrix.
        classes (list): List of class names.
        cm_file_name (str): Name of the file to save.
        save_folder (str): Folder to save the plot.
    """
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()  # Convert to NumPy if it's a tensor

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Compute class-wise accuracies
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    accuracy_str = "Accuracy: \n"
    for idx, class_name in enumerate(classes):
        accuracy_str += f" {class_name}: {round(class_accuracies[idx] * 100, 2)}%\n"

    plt.title('Confusion Matrix\n' + accuracy_str)

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the figure
    plt.savefig(os.path.join(save_folder, f"{cm_file_name}.png"))
    plt.close()
    return class_accuracies
# ----------------------------------------------------------------------------------------------------------
def create_image_mosaic(images_list, text_list, image_id, rows=4, cols=4, save_path='./',save_name='mosaic'):
    """
    Creates an image mosaic using Matplotlib, saves it, and returns it as a numpy array.

    Args:
    images_list (list): List of numpy array images.
    rows (int): Number of rows in the mosaic.
    cols (int): Number of columns in the mosaic.
    save_path (str): Path to save the mosaic image.

    Returns:
    numpy.ndarray: Image of the mosaic loaded as a numpy array for OpenCV.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(
        12, 12))  # Adjust the figure size as needed
    axes = axes.ravel()

    for idx, ax in enumerate(axes):
        if idx < len(images_list):
            ax.set_title(text_list[idx], fontsize=6)
            ax.imshow(images_list[idx], cmap='viridis' if len(
                images_list[idx].shape) == 2 else None)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()

    # Save the figure to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    # Close the plot to free memory
    plt.close(fig)

    # Convert buffer to OpenCV image format
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Optionaly save the image to disk
    if save_name and save_path:
        cv2.imwrite(os.path.join(save_path,save_name+"_"+str(image_id)+".png"), image)

    return image
# ----------------------------------------------------------------------------------------------------------
def plot_roc_and_calculate_auc(ground_truth, predicted_probabilities):
    """
    This function calculates the ROC curve and the AUC given a vector of true labels and predicted probabilities.

    Parameters:
    - ground_truth: numpy array with true labels (0s and 1s).
    - predicted_probabilities: numpy array with the probabilities of the positive class.

    Returns:
    - AUC score.
    """
    # Calcular puntos para la curva ROC
    fpr, tpr, _ = roc_curve(ground_truth, predicted_probabilities)
    # Calcular el AUC
    roc_auc = auc(fpr, tpr)

    # Graficar la curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc
# ----------------------------------------------------------------------------------------------------------