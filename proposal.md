# Project Proposal

## High-Level
Build an R-based tool using Keras to create a Convolutional Neural Network (CNN) for MNIST digit recognition, showing each CNN layer’s processing in 2D to demonstrate the model’s prediction process.

## Goals & Motivation
The project focuses on creating a CNN in R with Keras to recognize handwritten digits from the MNIST dataset and visualize each convolutional layer’s transformations in 2D. Users can draw a digit and see how the CNN processes it to predict the number.

The idea comes from our Data Visualization course, where we want to combine deep learning and visualization to explain how CNNs handle images. Using the MNIST dataset, a common benchmark for image recognition, we aim to make a clear, educational tool that shows how each CNN layer works and how its parameters affect results.

This project is interesting because it connects deep learning with visual tools, making neural network processes easier to understand. We will tweak CNN parameters like filter size, number of filters, stride, activation function, padding, loss function, optimizer (optional), pooling method, dropout rate (optional), and validation size to see their effects and show them visually.

We will use the MNIST dataset, available through Keras, to train and test the model. The visualization will focus on a simple setup with three convolutional layers and two fully connected layers, keeping it clear and focused on learning.

## Weekly Plan

| Week | Task |
|:---|:---|
| 1 | Team meeting: Set project scope, create GitHub repository, install Keras in R, and plan CNN architecture. |
| 2 | Load and preprocess MNIST dataset in R. Design initial CNN architecture with Keras (3 conv layers, 2 FC layers, 3x3 kernel, stride=2, padding=1). Research 2D visualization methods for CNN layers. |
| 3 | Build CNN model with ReLU activation, Binary Cross Entropy loss, Stochastic Gradient Descent optimizer (optional), and Max Pooling. Create scripts to extract and visualize layer outputs in 2D. Test model on MNIST data. |
| 4 | Improve visualization to show layer outputs for a user-drawn digit input. Add options to adjust parameters (e.g., filter size, number of filters). Document code and draft presentation. |
| 5 | Finalize visualization tool and model. Test with sample inputs and collect feedback. Create presentation slides. Record demo video showing the prediction process. |