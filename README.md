# CIFAR-10 Image Classification Project

## Overview
This project builds and trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. CIFAR-10 is a well-known dataset in computer vision, consisting of 60,000 32x32 color images in 10 different classes. The goal is to correctly identify images as one of the 10 classes. The project first used simple classification algorithms e.g. KNN, SVM, however, it resulted subpar accuracy (<50%). Then the project utilized one of the deep learning technique e.g. CNN. 

## Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. The classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Usage
The project is structured as a Jupyter Notebook which can be run in environments like Google Colab or Jupyter Lab. Follow the steps in the notebook to load the data, preprocess it, build the CNN model, train the model, and evaluate its performance.

## Model Architecture
The CNN model used in this project consists of the following layers:
- Convolutional Layer with 32 filters
- MaxPooling Layer
- Convolutional Layer with 64 filters
- MaxPooling Layer
- Flatten Layer
- Dense Layer with 64 neurons
- Output Dense Layer with 10 neurons (one for each class)

## Training
The model is trained for 10 epochs using the Adam optimizer and sparse_categorical_crossentropy as the loss function. Training and validation accuracy are printed at the end of each epoch to monitor the training progress.

## Evaluation
The model's performance is evaluated using the test set. The evaluation metric is accuracy.

## Results
The results section in the notebook shows the model's accuracy on the test set. It also includes plots of training and validation accuracy over epochs.

## Future Work
Future enhancements can include experimenting with different model architectures, hyperparameters, and using techniques like data augmentation to improve model performance.

## License
[MIT License](LICENSE)
