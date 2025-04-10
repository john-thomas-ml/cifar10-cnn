# cifar10-cnn

Welcome to **cifar10-cnn**, a minimalist deep learning project focused on classifying images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN). This project is designed as a learning exercise to explore image classification with TensorFlow and Keras, comparing a simple neural network (NN) with a CNN architecture.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is split into 50,000 training images and 10,000 test images. This project implements:
- A basic feedforward neural network (NN) as a baseline.
- A CNN to leverage spatial features in images, showcasing improved performance.

The code includes data preprocessing, model training, visualization of results, and evaluation metrics like accuracy, loss, classification reports, and confusion matrices.

## Features
- **Data Preprocessing**: Normalizes pixel values and visualizes sample images.
- **Model Comparison**: Trains and evaluates both a simple NN and a CNN.
- **Visualization**: Plots training progress, sample predictions, and confusion matrices.
- **Evaluation**: Provides detailed metrics including precision, recall, and F1-score.

## Results

### Neural Network (NN) Performance
The baseline NN was trained for 10 epochs with the following results:

- **Training Progress**:
  - Epoch 1: accuracy: 0.2815, loss: 1.9758
  - Epoch 5: accuracy: 0.4331, loss: 1.5774
  - Epoch 10: accuracy: 0.4664, loss: 1.4823
  - Time per epoch: ~1m 33s

- **Test Evaluation**:
  - Accuracy: 0.4605
  - Loss: 1.5270

- **Classification Report**:
  ```
  precision    recall  f1-score   support
  0       0.53      0.45      0.48     1000
  1       0.60      0.52      0.56     1000
  2       0.36      0.26      0.30     1000
  3       0.39      0.19      0.26     1000
  4       0.35      0.54      0.43     1000
  5       0.46      0.30      0.36     1000
  6       0.51      0.46      0.49     1000
  7       0.38      0.69      0.49     1000
  8       0.52      0.53      0.52     1000
  9       0.47      0.58      0.52     1000
  accuracy                          0.46     10000
  macro avg       0.46      0.45      0.44     10000
  weighted avg    0.46      0.45      0.44     10000
  ```

The NN achieves a modest accuracy of ~46%, indicating limited capability to capture spatial features in images.

### Convolutional Neural Network (CNN) Performance
The CNN was trained for 10 epochs with the following results:

- **Training Progress**:
  - Epoch 1: accuracy: 0.3424, loss: 1.7968
  - Epoch 5: accuracy: 0.6576, loss: 0.9797
  - Epoch 10: accuracy: 0.7317, loss: 0.7623
  - Time per epoch: ~1m 56s

- **Test Evaluation**:
  - Accuracy: 0.6744
  - Loss: 0.9771

- **Classification Report**:
  ```
  precision    recall  f1-score   support
  0       0.69      0.72      0.70     1000
  1       0.89      0.79      0.83     1000
  2       0.52      0.62      0.56     1000
  3       0.48      0.59      0.49     1000
  4       0.68      0.53      0.60     1000
  5       0.63      0.47      0.54     1000
  6       0.71      0.80      0.75     1000
  7       0.73      0.76      0.74     1000
  8      0.71      0.83      0.77     1000
  9      0.81      0.71      0.75     1000
  accuracy                          0.68     10000
  macro avg       0.68      0.68      0.67     10000
  weighted avg    0.68      0.68      0.67     10000
  ```

The CNN significantly outperforms the NN, achieving an accuracy of ~67.4%, demonstrating the effectiveness of convolutional layers for image classification.
