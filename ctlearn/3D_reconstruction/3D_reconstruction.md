# 3D Event Reconstruction

## 3D IACT Event Reconstruction from 2D multiple telescopic images pipeline

Reconstructing the event with CNNs: 

1. Detect important 2D features present in the images by Deep Convolutional Neural Network.
2. Match 2D points across different images captured by different cameras with backpropagation and gradient descent.
3. Epipolar geometry   
  3a. If both intrinsic and extrinsic camera parameters are known, reconstruct with projection matrices.   
  3b. If only the intrinsic parameters are known, normalize coordinates and calculate the essential matrix.   
  3c. If neither intrinsic nor extrinsic parameters are known, calculate the fundamental matrix.   


### To do list

- [ ] Modifications to the DataLoader class (and the data storage format) to handle truly 3D data. (see ctlearn/data_loader.py).
- [ ] Modifications to the DataProcessor class (see ctlearn/data_processing.py) to adapt the existing functionality (cropping, cleaning, normalization) to 3D inputs.
- [ ] Implementation of an entirely new module implementing a geometric 3D reconstruction method (from multiple shower images to a single 3D shower representation). A memo outlining some ideas for a proposed approach can be provided on request.
- [ ] Implementation of a 3D CNN model in Tensorflow in the supported model format (see ctlearn/default_models.py for examples).
- [ ] Implementation of any scripts required for visualization of results/verification of data/etc.


   