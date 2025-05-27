# from sklearn.manifold import TSNE
from cuml.manifold import TSNE

# from nets.models.ResNet import ResNet
from ctlearn.ctlearn_helper.common import read_configuration
from ctlearn import utils
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ctlearn.ctlearn_helper.CTlearnEnums import Task, Mode, EventType
from ctlearn.cli.run_model import Runner 
from ctlearn.utils.utils import (
    load_pickle,
    create_key_value_array

)
from ctlearn import (
    CTADataset,
    read_configuration,
    ModelHelper,
)
# from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import faiss 
import pickle 
from typing import Dict
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', depending on your setup

def plt_tsne(embeddings, pred_labels, gt_labels, class_energy, classes, plot_3d=False, post_fix="tsne"):
    """
    Visualize t-SNE embeddings with energy levels encoded by color, preserving class markers,
    and highlighting misclassified points.

    Parameters:
    - embeddings: t-SNE reduced feature vectors.
    - pred_labels: Predicted class labels.
    - gt_labels: Ground truth class labels.
    - class_energy: Energy values corresponding to each sample.
    - classes: List of class names.
    - plot_3d: Boolean to plot in 3D.
    - prefix: Prefix for the saved plot filename.
    """

    # Normalize energy values to [0, 1] for colormap mapping
    scaler = MinMaxScaler()
    norm_energy = scaler.fit_transform(class_energy.reshape(-1, 1)).flatten()

    # Choose a colormap
    cmap = matplotlib.colormaps['viridis']#cm.get_cmap('viridis')

    # Identify misclassified points
    misclassified = pred_labels != gt_labels

    # Initialize plot
    fig = plt.figure(figsize=(10, 8))
    if plot_3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    # Define markers for each class
    markers = ["o", "^", "s"]

    # Plot correctly classified points with energy-based colors
    for i, class_name in enumerate(classes):
        idx = (gt_labels == i) & ~misclassified
        if plot_3d:
            sc = ax.scatter(
                embeddings[idx, 0],
                embeddings[idx, 1],
                embeddings[idx, 2],
                color=cmap(norm_energy[idx]),
                marker=markers[i],
                label=f"Correct {class_name}",
                alpha=0.6
            )
        else:
            sc = ax.scatter(
                embeddings[idx, 0],
                embeddings[idx, 1],
                color=cmap(norm_energy[idx]),
                marker=markers[i],
                label=f"Correct {class_name}",
                alpha=0.6
            )

    # Plot misclassified points with a distinct marker and color
    if plot_3d:
        ax.scatter(
            embeddings[misclassified, 0],
            embeddings[misclassified, 1],
            embeddings[misclassified, 2],
            color="red",
            marker="x",
            label="Misclassified",
            alpha=0.1
        )
    else:
        ax.scatter(
            embeddings[misclassified, 0],
            embeddings[misclassified, 1],
            color="red",
            marker="x",
            label="Misclassified",
            alpha=0.1
        )

    # Set plot title and labels
    ax.legend(loc="best")
    ax.set_title("t-SNE Visualization with Energy Levels and Misclassifications")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if plot_3d:
        ax.set_zlabel("t-SNE 3")

    # Add colorbar to indicate energy levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=class_energy.min(), vmax=class_energy.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    
    cbar.set_label('Energy Levels')

    # Save and display plot
    plt.savefig(f'./tsne_{post_fix}.png')
    plt.show()

def main(use_pickle,reduce_similarity,plot_tsne):

    classes_list = ["gamma", "proton"]
    config_file_str = "./config/training_config_iaa_neutron_missing_analisys.yml"
    parameters = read_configuration(config_file_str)
    # file_name = parameters["data"]["train_gamma_proton"]
    file_name = "/storage/ctlearn_data/training/node_pickles/proton_proton_theta_16.087_az_108.090_runs1-416_train_275107.dl1.pickle"
    if not use_pickle:
        
        batch_size = 64 #parameters["hyp"]["batches"]
        num_workers = parameters["dataset"]["num_workers"]
        pin_memory = parameters["dataset"]["pin_memory"]
        persistent_workers = parameters["dataset"]["persistent_workers"]

        print("Data loaded...")

        validation_data =load_pickle(file_name) 

        factor = 1
        data_len = len(validation_data["data"])
        validation_data["data"] = validation_data["data"][0 : int(data_len / factor)]
        validation_data["true_shower_primary_id"] = validation_data[
            "true_shower_primary_id"
        ][0 : int(data_len / factor)]

        cta_ds_validation = CTADataset(
            pickle_data=validation_data, task=Task.type,mode=Mode.results, parameters=parameters, use_augmentation = False
            )
                
        data_loader = torch.utils.data.DataLoader(
            cta_ds_validation,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=cta_ds_validation.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        
    
        runner = Runner()
        model_net = runner.create_model(parameters["model"]["model_type"])

        device_str = parameters["arch"]["device"]
        device = torch.device(device_str)
        
        check_point_path = parameters["data"]["type_checkpoint"]
        
        model_net = ModelHelper.loadModel(
            model_net, "", check_point_path, mode=Mode.results , device_str=device_str
        )
            
        model_net.eval()

        class_feature_vector = []
        class_predictions_class = []
        class_gt_class = []
        class_energy = []
        class_hillas = []
        class_hillas_name = validation_data["hillas_name"]
        pbar = tqdm(total=len(data_loader), desc="DL2 conv", leave=True)

    
        for batch_idx, (features, labels) in enumerate(data_loader):

            imgs = features["image"].to(device).contiguous()
            peak_time = features["peak_time"].to(device).contiguous()
            classification_pred, energy_pred, direction_pred = model_net(
                imgs, peak_time
            )

            if len(features)==0:
                continue

            classification_pred_ = classification_pred[0]
            feature_vector = classification_pred[1].cpu().detach().numpy()
            predicted = torch.softmax(classification_pred_, dim=1)
            predicted_class = predicted.argmax(dim=1)
            predicted_class = predicted_class.cpu().detach().numpy()  

            labels_class = (labels["particletype"].int().to(device).contiguous())
            labels_class = labels_class.cpu().detach().numpy()  

            labels_energy = labels["energy"].cpu().detach().numpy()  

            hillas = features["hillas"]
            hillas = {
                key: tensor.cpu().detach().numpy() for key, tensor in hillas.items()
            }
            id = list(range(features["image"].shape[0]))

            hillas_vector = np.array(create_key_value_array(hillas, id)).T

            class_feature_vector.extend(feature_vector[:, :])
            class_predictions_class.extend(predicted_class[:])
            class_gt_class.extend(labels_class[:])
            energy = np.power(10,labels_energy[:])[:,0]
            class_energy.extend(energy)
            class_hillas.extend(hillas_vector)
            if batch_idx % 10 == 0:
                pbar.update(10)


        embeddings = np.array(list(class_feature_vector))
        pred_labels = np.array(list(class_predictions_class))
        gt_labels = np.array(list(class_gt_class))
        gt_energies = np.array(list(class_energy))
        class_hillas = np.array(list(class_hillas))

        data_dict ={"embeddings":embeddings,
                    "pred_labels":pred_labels,
                    "gt_labels":gt_labels,
                    "gt_energies":gt_energies,
                    "hillas_name":class_hillas_name,
                    "hillas":class_hillas}


        # Save pickle
        save_file_name= "./prediction_train.pickle"
        with open(save_file_name, "wb") as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Files saved ... ", save_file_name)
    

    save_file_name= "./prediction_train.pickle"
    # save_file_name= "./validation_data.pickle"

    prediction = load_pickle(save_file_name)
    print("pickle loaded...")
    embeddings = prediction["embeddings"]
    pred_labels = prediction["pred_labels"]
    gt_labels = prediction["gt_labels"]
    gt_energies = prediction["gt_energies"]
    hillas_name = prediction["hillas_name"]
    hillas = prediction["hillas"]
    if reduce_similarity:
        original_data =load_pickle(file_name)      
        SIMIL_THRS = 0.99

        #------------------------------------------------------------------------------------------
        # dimension = embeddings.shape[1]
        # nlist = 100  # Número de clusters (ajustar según dataset)
        # # Crear el índice con clusters
        # faiss.normalize_L2(embeddings) 
        # quantizer = faiss.IndexFlatIP(dimension)
        # index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        # # index.nprobe = 20
        # index.train(embeddings)  # Entrenar el índice de clustering
        # index.add(embeddings)  # Agregar embeddings
        # if not index.is_trained:
        #     print("Index NOT trained properly.")
        #     exit()
        # else:
        #     print("Index trained properly.")
        #------------------------------------------------------------------------------------------
        # dimension = embeddings.shape[1]
        # nlist = 100  # Número de clusters (ajustar según dataset)
        # # Crear el índice con clusters
        # faiss.normalize_L2(embeddings) 
        # # index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.index_factory(dimension, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)

        # index.nprobe = 20
        # index.train(embeddings)  # Entrenar el índice de clustering
        # index.add(embeddings)  # Agregar embeddings
        # if not index.is_trained:
        #     print("Index NOT trained properly.")
        #     exit()
        # else:
        #     print("Index trained properly.")            
        #------------------------------------------------------------------------------------------
        # dimension = embeddings.shape[1]
        # faiss.normalize_L2(embeddings) 
        # index = faiss.IndexHNSWFlat(dimension, 32)  # 32 vecinos en el grafo
        # index.hnsw.efConstruction = 60  # Controla la calidad del grafo (más grande = mejor recall)
        # index.hnsw.efSearch = 100  # Cuántos vecinos considerar en búsqueda

        # # Agregar embeddings
        # index.add(embeddings)
        #------------------------------------------------------------------------------------------
        #         
        #------------------------------------------------------------------------------------------
        # dimension = embeddings.shape[1]
        # index = faiss.IndexFlatIP(dimension)  # Index con producto interno (similaridad coseno)
        # index = faiss.IndexIDMap(index)
        # faiss.normalize_L2(embeddings)  # Normalizar embeddings para similitud del coseno
        # index.add(embeddings)  # Agregar embeddings al índice de Faiss
        #------------------------------------------------------------------------------------------
        # print("Generating index.")
        # embeddings = np.array(embeddings).astype('float32')
        # faiss.normalize_L2(embeddings) 
        # dimension = embeddings.shape[1]
        
        # nlist = 50 #100  # Número de clusters (ajustar según dataset)
        # # Step 1: Create the CPU index
        # quantizer = faiss.IndexFlatIP(dimension)
        # cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # # Step 2: Train on CPU
        # cpu_index.train(embeddings)

        # # Step 3: Move to GPU
        # res = faiss.StandardGpuResources()  # Use default options
        # index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 0 is the GPU id
        # index.nprobe = 80
        # # Step 4: Add embeddings to GPU index
        # index.add(embeddings)
        # if not index.is_trained:
        #     print("Index NOT trained properly.")
        #     exit()
        # else:
        #     print("Index trained properly.")


        print("Generating index.")
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        nlist = 50  # Número de clusters
        quantizer = faiss.IndexFlatIP(dimension)
        cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # Entrenar índice en CPU
        cpu_index.train(embeddings)
        num_gpus = faiss.get_num_gpus()
        print("GPUs detected:",num_gpus)
        # Crear recursos para múltiples GPUs
        gpu_resources = [faiss.StandardGpuResources() for _ in range(num_gpus)]   

        # Distribuir el índice entrenado a múltiples GPUs
        gpu_indices = [
            faiss.index_cpu_to_gpu(gpu_resources[i], i, cpu_index) 
            for i in range(2)
        ]

        # Combinar índices GPU en un índice shard
        index = faiss.IndexShards(dimension, True, False)
        for sub_index in gpu_indices:
            index.add_shard(sub_index)

        index.nprobe = 80

        # Agregar embeddings al índice distribuido (se distribuyen automáticamente entre GPUs)
        index.add(embeddings)

        if not index.is_trained:
            print("Index NOT trained properly.")
            exit()
        else:
            print("Index trained properly.")

        #------------------------------------------------------------------------------------------

        # print("Generating index.")
        # embeddings = np.array(embeddings).astype('float32')
        # faiss.normalize_L2(embeddings) 
        # dimension = embeddings.shape[1]

        # nlist = 50#100  # Número de clusters (ajustar según dataset)

        # # Paso 1: Crear el índice en CPU
        # quantizer = faiss.IndexFlatIP(dimension)
        # index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # # Paso 2: Entrenar el índice (CPU)
        # index.train(embeddings)

        # index.nprobe = 80

        # # Paso 3: Añadir los embeddings al índice CPU
        # index.add(embeddings)

        # # Verificación del entrenamiento
        # if not index.is_trained:
        #     print("Index NOT trained properly.")
        #     exit()
        # else:
        #     print("Index trained properly.")        
        #------------------------------------------------------------------------------------------

        print("Looking for duplicates: Search...")
        # D, I = index.search(embeddings, k=3)  # Search for the k nearest 
        # lims, D, I = index.range_search(embeddings, SIMIL_THRS)
        # Parameters
        SIMIL_THRS = 0.99
        k_search_default= 10
        k_search_first_bins = 45 #25
        similarity_default = 0.95
        similarity_first_bins = 0.90
        k_bin=11
        cut_off_leakage_intensity = 0.2
        cut_off_intensity = 50
        bins = np.logspace(np.log10(2.51e-02), 2, 19)
        bin_indices = np.digitize(gt_energies, bins)
        num_bins = len(bins)  # should be 18 in your case

        # Create array with default value 0.99
        simil_thresholds = np.full(num_bins+1, similarity_default)
        k_search = np.full(num_bins+1, k_search_default,dtype=int)
        k_search[:k_bin] = k_search_first_bins
        # Set first 5 bins to 0.95
        simil_thresholds[:k_bin] = similarity_first_bins
        # === Deduplication within bins ===
        keep_image = set()
        deleted = set()
        deleted_bin = set()
        seen = set()

        unique_bins = np.unique(bin_indices)
        print(f"Processing {len(unique_bins)} bins...")

        deleted_list=[]
        for b in tqdm(unique_bins):
            # Get indices in the current bin
            bin_mask = (bin_indices == b)
            bin_ids = np.where(bin_mask)[0]

            if len(bin_ids) < 2:
                continue  # Skip bins with fewer than 2 items

            # Extract embeddings for this bin
            bin_embeddings = embeddings[bin_ids]
            print(f"Processing bin {b} of {len(unique_bins)} ")
            # Perform FAISS k-NN search
            D, I = index.search(bin_embeddings, k=int(k_search[b]))
            # D, I = index.search(bin_embeddings, k=k_search_default)


            for i in range(len(bin_ids)):
                query_idx = bin_ids[i]

                if query_idx in deleted:
                    continue
                
                if hillas[query_idx][hillas_name["leakage_pixels_width_2"]]>cut_off_leakage_intensity or hillas[query_idx][hillas_name["hillas_intensity"]]<cut_off_intensity:
                    deleted.add(query_idx)
                    deleted_bin.add(query_idx)
                    continue
                else:
                    keep_image.add(query_idx)
                    seen.add(query_idx)

                for j in range(1, k_search[b]):  # Skip j=0 (self-match)
                    neighbor_idx_local = I[i][j]
                    similarity = D[i][j]

                    if neighbor_idx_local >= len(bin_ids):
                        continue  # Index out of bounds (can happen if few items in bin)

                    neighbor_idx = bin_ids[neighbor_idx_local]

                    if neighbor_idx == query_idx:
                        continue

                    if similarity >= simil_thresholds[b] and neighbor_idx not in seen:
                        deleted.add(neighbor_idx)
                        deleted_bin.add(neighbor_idx)
                        seen.add(neighbor_idx)

            deleted_list.append(deleted_bin)
            deleted_bin = set()
            seen = set()
        print(f"Final samples kept: {len(keep_image)}")
        print(f"Samples removed as similar: {len(deleted)}")

        # Count number of deletions per bin
        deleted_counts = [len(bin_deleted) for bin_deleted in deleted_list]
        kept_indices = sorted(keep_image)

        #------------------------------------------------------------------------------------------
        # Save data 
        # original_data["data"]=original_data["data"][kept_indices]
        original_data["data"] = [original_data["data"][i] for i in kept_indices]
        save_file_name= "./train_reduced_data.pickle"
        
        with open(save_file_name, "wb") as handle:
            pickle.dump(original_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #------------------------------------------------------------------------------------------
        # Get indices for each class in the ground truth
        gamma_indices = np.where(gt_labels == 0)[0]
        proton_indices = np.where(gt_labels == 1)[0]

        # Gamma accuracy: correct predictions among all gamma ground truths
        gamma_accuracy = np.mean(pred_labels[gamma_indices] == 0)

        # Proton accuracy: correct predictions among all proton ground truths
        proton_accuracy = np.mean(pred_labels[proton_indices] == 1)

        overall_accuracy = np.mean(gt_labels == pred_labels)

        print(f"Accuracy Before: {overall_accuracy}")
        print(f"Accuracy Gamma Before: {gamma_accuracy}")
        print(f"Accuracy Proton Before: {proton_accuracy}")      
        print(f"Num of Gammas Before: {len(gamma_indices)}")         
        print(f"Num of Protons Before: {len(proton_indices)}")       
        #------------------------------------------------------------------------------------------
        # Get indices for each class in the ground truth
        gamma_indices = np.where(gt_labels[kept_indices] == 0)[0]
        proton_indices = np.where(gt_labels[kept_indices] == 1)[0]

        # Gamma accuracy: correct predictions among all gamma ground truths
        gamma_accuracy = np.mean(pred_labels[kept_indices][gamma_indices] == 0)

        # Proton accuracy: correct predictions among all proton ground truths
        proton_accuracy = np.mean(pred_labels[kept_indices][proton_indices] == 1)

        overall_accuracy = np.mean(gt_labels[kept_indices] == pred_labels[kept_indices])

        print(f"Accuracy After: {overall_accuracy}")
        print(f"Accuracy Gamma After: {gamma_accuracy}")
        print(f"Accuracy Proton After: {proton_accuracy}")      
        print(f"Num of Gammas After: {len(gamma_indices)}")         
        print(f"Num of Protons After: {len(proton_indices)}")       
        #------------------------------------------------------------------------------------------        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(deleted_counts)), deleted_counts)
        plt.xlabel("Bin Index")
        plt.ylabel("Number of Deleted Samples")
        plt.title("Deleted Samples per Energy Bin")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./histogram_deleted_{k_search_first_bins}_{k_search_default}.png') 
        # plt.show()       

        #------------------------------------------------------------------------------------------
        # Create bin labels (log-scale bins)
        # Ensure we have 18 bins
        # Bin labels: 18 labels for 18 bins
        # num_bins = len(bins)         
        # bin_labels = [f"{bins[i]:.2e}-{bins[i+1]:.2e}" for i in range(num_bins-1)]
        # deleted_counts = [len(s) for s in deleted_list]
        # x = list(range(num_bins))

        # plt.figure(figsize=(12, 6))
        # plt.bar(x, deleted_counts)
        # plt.xticks(ticks=x, labels=bin_labels, rotation=45, ha='right')
        # plt.xlabel("Energy Bin")
        # plt.ylabel("Number of Deleted Samples")
        # plt.title("Deleted Samples per Energy Bin")
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.tight_layout()
        # plt.show()
        #------------------------------------------------------------------------------------------
        # Convert sets to sorted lists (for consistent indexing)
        deleted_indices = sorted(deleted)
        kept_indices = sorted(keep_image)


        # Get energy values
        energies_before = gt_energies
        energies_after = gt_energies[kept_indices]

        # Create log-spaced bins for histogram (same as your binning)
        hist_bins = np.logspace(np.log10(2.51e-02), 2, 30)

        plt.figure(figsize=(12, 6))

        # Plot before as outline
        plt.hist(energies_before, bins=hist_bins, histtype='step', label='Before Filtering', color='gray', linewidth=1.5)

        # Plot after as filled
        plt.hist(energies_after, bins=hist_bins, alpha=0.6, label='After Filtering', color='green')

        plt.xscale('log')
        plt.xlabel('Energy')
        plt.ylabel('Number of Samples')
        plt.title('Energy Histogram Before and After Filtering')
        plt.legend()
        plt.grid(True, which='both', ls='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'./histogram_energy_before_after_{k_search_first_bins}_{k_search_default}.png')   
        # plt.show()      
        #------------------------------------------------------------------------------------------

        # flat_similarities = D[:, 1:].flatten()  # excluye el self-match
        # plt.hist(flat_similarities, bins=100)
        # plt.axvline(SIMIL_THRS, color='red', linestyle='--')
        # plt.title("Distribución de Similitudes (excluyendo self-match)")
        # plt.xlabel("Similitud")
        # plt.ylabel("Frecuencia")
        # plt.show()


        # EPS = 1e-5
        # keep_image = set()
        # deleted = set()
        # seen = set()

        # for i in tqdm(range(len(embeddings))):
        #     if i in deleted:
        #         continue

        #     keep_image.add(i)
        #     if gt_energies[i]<1:
        #         for j in range(1, len(I[i])):
        #             neighbor_idx = I[i][j]
        #             similarity = D[i][j]

        #             if similarity < (SIMIL_THRS - EPS):
        #                 continue  # no lo consideres duplicado

        #             if neighbor_idx not in keep_image:
        #                 deleted.add(neighbor_idx)
        #------------------------------------------------------------------------------------------
        # keep_image = set()
        # deleted = set()
        # print("Search Done")
        # for i in tqdm(range(len(embeddings))):
        #     if i in deleted:
        #         continue  # Si ya está marcada como duplicada, ignorarla
        #     keep_image.add(i)  # Mantener esta imagen
        #     start_idx = lims[i]
        #     end_idx = lims[i + 1]
        #     for j in range(start_idx, end_idx):
        #         neighbor_idx = I[j]
        #         if neighbor_idx != i:  # Evitar la propia imagen
        #             deleted.add(neighbor_idx)  # Marcar como duplicada
        #             # print(f"Eliminando duplicado: {image_paths[neighbor_idx]} (ID {image_ids[neighbor_idx]})")


        print(f"keeped: {len(keep_image)}")
        print(f"deleted: {len(deleted)}")
        embeddings = embeddings[list(keep_image)]
        pred_labels = pred_labels[list(keep_image)]
        gt_labels =  gt_labels[list(keep_image)]
        gt_energies = gt_energies[list(keep_image)]

    if plot_tsne:
        # energy_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # energy_threshold = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        print("Plotting t-sne")

        energy_threshold=[0]
        for energy_thr in energy_threshold:
            embeddings_ = embeddings[gt_energies>energy_thr]
            pred_labels_ = pred_labels[gt_energies>energy_thr]
            gt_labels_ = gt_labels[gt_energies>energy_thr]
            gt_energies_ = gt_energies[gt_energies>energy_thr]


            print(f"generating thr:{energy_threshold}")
            plt_tsne(embeddings_, pred_labels_, gt_labels_, gt_energies_, classes_list, post_fix="v4_"+str(energy_thr))
 

if __name__ == "__main__":

    use_pickle = True
    reduce_similarity = True
    plot_tsne=True

    main(use_pickle,reduce_similarity,plot_tsne)
