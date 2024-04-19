# VEBI-PyTorch-Implementation

This repositoy provides the PyTorch implementation of a post-hoc model interpretation method titled: "Visual Explanation by Interpretation: Improving Visual Feedback Capabilities of Deep Neural Networks" (Oramas et al. ICLR 2019).

## Description of files are as follows

VEBI has two phases: (1) Collecting activation maps (2) Identifying the relevant internal units.

1- collect_all_actmaps.py: Pushes images through the models and collect the activation maps of each units per layer.

2- identifying_relevant_features.py: Implements mu-LASSO problem to identify critical internal relevant units.

3- generate_heatmap.py & generate_viz.py: In order to provide visualizations from identified units, these two files generate heatmaps per identified units for each input image, followed by superimposing on the input image. Examples of visualization generated by the implemented code are as follows.

## Visualization results on ClebA dataset.

![Screenshot from 2024-04-19 10-03-51](https://github.com/hamedbehzadi/VEBI-PyTorch-Implementation/assets/45251957/55d6038a-8bac-472b-8bfc-678945c60b9a)




## Visualization results on CUB dataset.

![Screenshot from 2024-04-19 10-05-10](https://github.com/hamedbehzadi/VEBI-PyTorch-Implementation/assets/45251957/722bcc5e-f33f-408a-b6ff-966872b5eec7)


## Setup

'''

'''






