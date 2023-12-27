# Best Artworks of All Time - GANs Project
## Overview
This repository contains code for a Generative Adversarial Network (GAN) project focused on generating paintings from influential artists throughout history. The project utilizes a dataset comprising artworks from 50 of the most influential artists, exploring the generation of new, artificial images based on their styles.

## Dataset
* The dataset used in this project includes information about artists along with a collection of resized images for training the GAN model.
* It can be downloaded from [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time).

## Libraries Used
* PyTorch
* TorchVision
* Matplotlib
* tqdm
* Jovian

## Code Structure
### Download and Exploration of Dataset: 
* Code for downloading and exploring the dataset using `opendatasets`.
### Data Preprocessing: 
* Preparing the dataset for training by resizing images and creating PyTorch datasets and data loaders.
### Defining the Models: 
* Creating the discriminator and generator neural networks using PyTorch.
### Training Loop: 
* Training the GAN model with the discriminator and generator in tandem, optimizing their respective losses.
### Visualization: 
* Displaying generated images, loss curves, and scores to monitor training progress.
### Saving Checkpoints: 
* Saving trained model checkpoints for future use.

## Training
* The training process involved multiple epochs with different learning rates, carefully balancing the training of the generator and discriminator.
The model was periodically saved during training to capture intermediate progress.

## Results
Visualizations of loss curves and scores are provided to track the performance and convergence of the GAN model.
Sample generated images are displayed at various stages of training to demonstrate the model's progression.
