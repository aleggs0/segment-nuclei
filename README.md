# segment-nuclei
This repository contains work from Alex's summer project on segmentation for microscopy at the University of Cambridge, 2024. With thanks to: Cambridge Advanced Imaging Centre, Department of Physiology, Development and Neuroscience; Light Microscopy Facility, MRC Laboratory of Molecular Biology; CMS Bursary.

About our dataset:
- mouse embryos, preimplantation
- fluorescent stain on nuclei
- light sheet microscopy for 3D images
- anisotropy factor of 11.5 (voxel size 0.174*2*2 micron)
- roughly 100 timepoints for each embryo
- one embryo was hand-labelled with nuclei segmentation
- a further eleven embryos had nuclei segmentation and tracking performed automatically and then manually corrected in previous work by my supervisors (Karsa et al, manuscript in preparation)

## Contents
1. A Cellpose (Stringer et al, 2021) model retrained on mouse embryo data.
  We found that although the pretrained nuclei model performed well on xy slices, it did poorly on xz and yz slices, even after rescaling to correct for anisotropy. To address this, our model is retrained from the nuclei model, using slices taken in not only the xy plane, but also the xz and yz planes, where the images were rescaled to correct for the anisotropy. Results on a test set taken from other mouse embryos show a noticeable improvement from using either stitched or extended Cellpose with the pretrained nuclei model.

2. Code for a neural network to detect cell division
   Nuclei show a change in appearance just before they divide. Detecting these nuclei has the potential to assist cell-tracking algorithms. We trained a neural network on x-y planes of our data to detect cell divisions. The next step in this direction would be to perform hyperparameter tuning and improve the flexibility of the algorithm (to allow for tiling in the predictions.

3. Some snippets of code for data processing, in the tools folder. Some of this was used as preprocessing steps for data in (1) and (2). In addition, embryo_seg_playgroun.ipynb has notes for how we segmented the embryos themselves in images containing multiple embryos in 4D (3D space plus time).

## References
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106.

Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature methods, 1-8.

Eschweiler, D., Smith, R. S., Stegmaier J. (2021) "Robust 3D Cell Segmentation: Extending the View of Cellpose", arXiv:2105.00794

Karsa, A., Boulanger, J., Abdelbaki, A., Niakan, K. K., Muresan, L. (Manuscript in preparation). A novel pipeline for robust and accurate, automated 3D nucleus segmentation and tracking in light-sheet microscopy images
