# segment-cell-division
A neural network to detect cell division

Run train.py to train; then run get_predictions.py to generate predictions for images in a folder (warning: images must all be of same size for get_predictions)

About our dataset:
- mouse embryos, preimplantation
- fluorescent stain on nuclei
- light sheet microscopy
- each nucleus has diameter approx. 30 pixels
- for ground truth, cells were labelled as 1 if about to divide, between 0 and 1 if dividing soon, and 0 if not dividing soon
- training data has ~500 instances of cell divisions captured in ~8000 2D images

To set up the environment, we used
```
conda create -n imaging python=3.10
conda activate imaging
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm tifffile scipy scikit-image numpy matplotlib pandas opencv-python
```

This code is adapted from code by Aladdin Pearson presented in https://www.youtube.com/watch?v=IHq1t7NxS8k&t=2786s and available at https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet

The NN architecture is the U-Net for the Cellpose algorithm presented in Stringer et al (2021), available at https://github.com/MouseLand/cellpose/tree/main/cellpose
