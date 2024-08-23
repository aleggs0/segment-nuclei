# Training a model for cell division

- Training data: 3D volumes of 11 embryos with 120 frames each. We first segmented the embryos (using the process described in ``tools/embryo_seg_playground.pynb``). We masked out any signal that was not part of the embryo and cropped the images down to the size of the embryos. For each cropped image, now containing only one embryo, we downsampled by a factor of 2 in the x and y directions (so that the typical diameter of a nucleus was ~35 pixels) and then took only those 2D slices in the x-y plane that contained a nonzero label. (Thus our approach is fully 2D.) The ground truth segmentation of each embryo had been obtained by Karsa et al. (manuscript in prepartion) who applied an automatic segmentation and tracking algorithm and then performed manual correction. From this, we obtain labels for cell divisions by setting voxels to 1 if its nucleus is about to divide, 0 if not dividing soon or not part of the cell, and between 0 and 1 if dividing soon (explicitly, we used values 0.8, 0.6, 0.4, 0.2 as the time to division became progressively longer). Altogether, the training data has ~500 instances of cell divisions captured in ~6000 2D images.

- Directory structure: within a data directory, there are subdirectories with names ``img``, ``val_img``, ``div_lbl``, ``val_div_lbl``.  The intensity files are saved with ``_img.tif`` suffices and corresponding labels are saved with ``_div.tif`` suffices.

- To set up the environment, we used
  ```
  conda create -n imaging python=3.10
  conda activate imaging
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install tqdm tifffile scipy scikit-image numpy matplotlib pandas opencv-python
  ```
- Run ``train.py`` to train; then run ``get_predictions.py`` to generate predictions for images in a folder (warning: images must all be of same size to generate predictions as tiling has not yet been implemented).

## Notes
 - This code is adapted from code by Aladdin Pearson presented in https://www.youtube.com/watch?v=IHq1t7NxS8k&t=2786s and available at https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet
 - The NN architecture is the U-Net for the Cellpose algorithm presented in Stringer et al (2021), available at https://github.com/MouseLand/cellpose/tree/main/cellpose
 - The next steps in this direction would be to perform hyperparameter tuning and improve the flexibility of the software to allow for tiling of test images. After this, we would want to incorporate this tool into a tracking algorithm.
