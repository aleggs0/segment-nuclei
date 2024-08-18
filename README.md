# segment-nuclei
This repository contains outputs from Alex's summer project on nucleus segmentation at the University of Cambridge, 2024. With thanks to: Cambridge Advanced Imaging Centre, Department of Physiology, Development and Neuroscience; Light Microscopy Facility, MRC Laboratory of Molecular Biology; CMS Bursary.

1. A Cellpose (Stringer et al, 2021) model retrained on mouse embryo data.
  We found that although the pretrained nuclei model performed well on xy slices, it did poorly on xz and yz slices, even after rescaling to correct for anisotropy. To address this, our model is retrained from the nuclei model, using slices taken in not only the xy plane, but also the xz and yz planes, where the images were rescaled to correct for an anisotropy factor of 11.5 (with diam_mean=30). The images are of mouse embryos in the preimplantation stage INSERT PHOTO HERE!
  To use this, (i) download ``models/nuclei_3d``, (ii) install [Cellpose](https://github.com/MouseLand/cellpose) in your python environment, (iii) in the command line with your environment activated, EITHER load the model using ``python -m cellpose --add_model /path/to/model/nuclei_3d`` and obtain predicions on your data using e.g. ``python -m cellpose --dir /path/to/imgs/ --pretrained_model nuclei_3d --do_3D --diameter 30 --anisotropy 11.5 --save_tif`` (using the diameter and anisotropy of your own data) OR run the Cellpose GUI with ``python -m cellpose --Zstack``, and use the GUI to load and apply the model. Note we do not include a size model. INSERT PHOTO HERE!

2. A neural network to detect cell division
Uplo

Run train.py to train; then run get_predictions.py to generate predictions for images in a folder (warning: images must all be of same size for get_predictions)

3. Some snippets of code for data processing, in the FOLDERNAME folder. Some of this was used as preprocessing steps for data in (1) and (2). In addition, embryo_seg_playgroun.ipynb has notes for how we segmented the embryos themselves in images containing multiple embryos in 4D (3D space plus time) INSERT PHOTOS

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
