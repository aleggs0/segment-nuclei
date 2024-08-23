# Training a model for cell division

- Training data: 3D volumes of 11 embryos with 120 frames each. This originally had anisotropy factor 11.5, with a typical diameter of ~70 pixels in the x and y directions. By downsampling by a factor of 2 in the x and y directions and upsampling by a factor of 5.75 in the z-direction (by linear interpolation), we had isotropic images and a typical diameter of 35. We took slices in the x-y, x-z and y-z planes: we took every x-y plane, and sampled every twelfth x-z and y-z plane. We used only planes containing at least three instances of nuclei, giving roughly 8000 planes of training data in total. These were all saved as .tif files in one folder (see tools/resize_dataset_original_embryo.py).
Training data: each nucleus has diameter approx. 30 pixels- for ground truth, cells were labelled as 1 if about to divide, between 0 and 1 if dividing soon, and 0 if not dividing soon- training data has ~500 instances of cell divisions captured in ~8000 2D images
![image](https://github.com/user-attachments/assets/e43b2474-f0c0-4a17-a821-24ae90b3770a)


Run train.py to train; then run get_predictions.py to generate predictions for images in a folder (warning: images must all be of same size for get_predictions)
To set up the environment, we used```conda create -n imaging python=3.10conda activate imagingpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121pip install tqdm tifffile scipy scikit-image numpy matplotlib pandas opencv-python```This code is adapted from code by Aladdin Pearson presented in https://www.youtube.com/watch?v=IHq1t7NxS8k&t=2786s and available at https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unetThe NN architecture is the U-Net for the Cellpose algorithm presented in Stringer et al (2021), available at https://github.com/MouseLand/cellpose/tree/main/cellpose
![image](https://github.com/user-attachments/assets/144c7acd-9141-4ded-89f6-6f66599db870)


The next steps in this direction would be to perform hyperparameter tuning and improve the flexibility of the software to allow for tiling of test images.
