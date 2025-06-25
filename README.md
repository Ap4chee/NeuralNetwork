Neural Network for 2x2 Image Pattern Classification
This project implements a two-layer feedforward neural network for precise pattern classification in miniature 2x2 images. The network is designed for efficiency and stability in recognizing distinct image patterns.

Project Overview
The core objective is to classify four specific 2x2 image patterns: Uniform, Horizontal, Vertical, and Diagonal. The system leverages synthetic data generation and various regularization techniques to achieve high accuracy.


Network Architecture
The network consists of:

An input layer with 4 neurons, each corresponding to a pixel in the 2x2 image.
A hidden layer with 8 neurons, utilizing a sigmoid activation function.
An output layer with 4 neurons, representing the unique pattern classes, with a softmax activation function for probabilistic outputs.
Weights are initialized using the Xavier method, drawing from a normal distribution based on layer size, while biases are set to zero.

Training Data
Synthetic training data is generated, with 500 to 1000 samples per class (defaulting to 1000), totaling 2000-4000 samples. Each sample has 4 features representing pixel brightness values. Specific rules govern the generation for each class to ensure diversity and clear differentiation, involving uniform distributions for base brightness and added Gaussian noise. Preprocessing includes normalization of pixel values to [0,1], random permutation of samples, and RGB to brightness conversion if needed.




Learning Algorithm Implementation
The learning algorithm is encapsulated within a NeuralNetwork class. Training involves forward and backward passes, cost function calculation, and gradient application over a set number of epochs (default 2000). Gradient clipping is applied to prevent exploding gradients, scaling down gradients if their norm exceeds 1.0. Early stopping is implemented to prevent overfitting; training halts if accuracy doesn't improve for 100 consecutive epochs after reaching 95% accuracy. Numerical stability is also enhanced for sigmoid and softmax functions. During training, metrics like loss, accuracy, and gradient norm are monitored, and a confusion matrix is generated periodically.



Results
The network consistently achieves 95-98% accuracy, with the loss function rapidly decreasing from around 1.4 to 0.08. Training time is very short, typically 1-5 seconds, and the network shows quick convergence, often stabilizing within 500-1000 epochs. The early stopping mechanism further optimizes training time. This makes the designed neural network an effective, stable, and reliable solution for 2x2 image pattern classification.
