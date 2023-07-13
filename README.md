[![License: GNU GPLv3](https://img.shields.io/badge/License-GNU_GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0/)
# Markov-based Neural Networks for Heart Sound Segmentation: Using Domain Knowledge in a principled way
We unify the statistical and data-driven solutions, by introducing *Markov-based Neural Networks* (MNNs), a hybrid end-to-end framework that exploits Markov models as statistical inductive biases for an Artificial Neural Network (ANN) discriminator. This repository provides the source code to replicate an MNN leveraging a simple one-dimensional Convolutional ANN that significantly outperforms two recent purely data-driven solutions for the task of fundamental heart sound segmentation in two publicly available datasets: PhysioNet 2016 (Sensitivity: $0.947 \pm 0.02$; Positive Predictive Value: $0.937 \pm 0.025$) and CirCor DigiScope 2022 (Sensitivity: $0.950 \pm 0.008$; Positive Predictive Value: $0.943 \pm 0.012$). 

We also introduce a gradient-based unsupervised fine-tuning algorithm that effectively makes the MNN adaptive to unseen datum sampled from unknown distributions. We show that a pre-trained MNN can learn to fit an entirely new dataset in an unsupervised fashion with remarkable gains in performance.
## Requisites
The user should have Python 3.8.10 installed.

## Requirements
One should create a python or conda virtual environment using the provided requirements.txt file

## Contribution
Each feature should be preferably implemented in a short-lived branch.
Commit messages should should follow these rules:
  * Seperate subject from body by an empty line
  * Subject should be no longer than 50 characters
  * The body should be no longer than 75 characters as often as possible
  * Use imperitive mood in title/subject of the commit
  * New features should have their purpose clearly stated in the body
  * Changes in packages should be reflected in requirements.txt, following lexographic order
