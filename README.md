# NeuralNetwork
Neural Network for 2x2 Image Pattern Classification
This repository contains the implementation of a two-layer feedforward neural network designed for precise pattern classification in miniature 2x2 images. The network combines simplicity with efficiency, enabling fast and accurate feature recognition.

Table of Contents
Project Overview
Network Architecture
Training Data
Learning Algorithm Implementation
Results
Project Overview
This project focuses on building and training a neural network to classify four distinct patterns in 2x2 pixel images: Uniform, Horizontal, Vertical, and Diagonal. The network is designed to be efficient and stable, leveraging synthetic data generation and various regularization techniques.

Network Architecture
The neural network is a two-layer feedforward design:

Input Layer: 4 neurons, directly corresponding to each pixel of a 2x2 image.
Hidden Layer: 8 neurons with a sigmoid activation function. This number of neurons was found to be optimal for the task. 

Sigmoid function:
sigma(x)=
frac11+e 
−x
  
Output Layer: 4 neurons, representing the four unique pattern classes. A softmax activation function transforms raw outputs into probabilities for multi-class classification. 

Softmax function: softmax(z_i)=
frace 
(z_i−max(z))
 sum_je 
(z_j−max(z))
  
Weight Initialization:
Weights are initialized using the Xavier method to mitigate vanishing or exploding gradients.

First layer weights (W1): Sampled from a normal distribution N(0,
sqrt2/textinput_size), where
textinput_size=4.
Second layer weights (W2): Sampled from a normal distribution N(0,
sqrt2/texthidden_size), where
texthidden_size=8.
Biases: All biases are initialized to zero.
Pattern Classes:
The network is trained to recognize four specific 2x2 image patterns:

Class 0 (Uniform): All pixels have very similar brightness values.
Class 1 (Horizontal): Characterized by a clear brightness difference between the top and bottom rows of pixels.
Class 2 (Vertical): Features a noticeable brightness difference between the left and right columns of pixels.
Class 3 (Diagonal): Brightness differences are most pronounced along the diagonals.
Training Data
Synthetic data was generated for training:

Number of Samples: 500 to 1000 samples per class (defaulting to 1000), resulting in a total of 2000-4000 training samples.
Features: Each sample consists of 4 features, corresponding to the brightness values of the 2x2 image pixels.
Generation Rules for Each Class:

Class 0 (Uniform): Base brightness is sampled from U(0.1,0.9), with a small Gaussian noise N(0,0.05) added to all pixels.
Class 1 (Horizontal): Top and bottom brightness values are sampled from U(0.1,0.9). A critical difference of at least 0.2 between top and bottom brightness is ensured (∣
texttop−
textbottom∣
ge0.2). Noise: N(0,0.03).

Class 2 (Vertical): Left and right brightness values are sampled from U(0.1,0.9). A critical difference of at least 0.2 between left and right brightness is ensured (∣
textleft−
textright∣
ge0.2). Noise: N(0,0.03).
Class 3 (Diagonal): Diagonal 1 (pixels 0,3) and Diagonal 2 (pixels 1,2) brightness values are sampled from U(0.1,0.9). A critical difference of at least 0.2 between diagonals is ensured (∣
textdiagonal1−
textdiagonal2∣
ge0.2). Noise: N(0,0.03).

Preprocessing:

Normalization: All pixel values are normalized to the range [0, 1].
Random Permutation: Samples are randomly shuffled to prevent the network from learning data sequences.
RGB to Brightness Conversion: For RGB input, brightness is calculated as 0.299
timesR+0.587
timesG+0.114
timesB.
Learning Algorithm Implementation
The learning algorithm is implemented within a modular NeuralNetwork class.

Training Algorithm:

Initialization: Weights are set using Xavier initialization, and biases are zeroed.
Training Loop: The network trains for a specified number of epochs (default 2000). In each epoch: 
Forward Pass: Training data is fed through the network to get predictions.
Cost Function Calculation: The cross-entropy loss between predictions and true labels is calculated.
Backward Pass: Gradients are computed and applied to update weights, using gradient clipping.
Accuracy Calculation: Current classification accuracy is monitored.
Gradient Clipping: Applied to prevent exploding gradients. If the gradient norm exceeds 1.0, gradients are scaled down.

Early Stopping: To prevent overfitting, training stops if accuracy does not improve for 100 consecutive epochs, provided accuracy has already exceeded 95%.

Monitoring: Key metrics like loss, accuracy, and gradient norm are displayed regularly. A confusion matrix is generated every 20% of epochs for detailed performance analysis.


Regularization and Stabilization Techniques:

Gradient Clipping: Limits gradient norm to 1.0 to prevent exploding gradients.
Numerical Stability (Sigmoid): Sigmoid input values are clipped to [-500, 500] for numerical stability.
Softmax Stability: Softmax implementation includes a shift by max(z) to avoid numerical issues with exponentials.
Early Stopping: Automatically halts training when the network stops improving, minimizing overfitting risk.
Results
The network, despite a reduced hidden layer size (8 neurons), achieved high effectiveness for 2x2 image pattern classification.

Typical Results after 2000 Epochs:

Accuracy: Consistently 95-98%.
Loss: Rapidly decreases from approximately 1.4 to 0.08.
Training Time: Extremely short, typically 1-5 seconds, due to the efficient architecture.
Convergence: Rapid convergence, with stable results often achieved within 500-1000 epochs. Early stopping further optimizes this process.

Summary:
The designed neural network provides an effective, stable, and reliable solution for 2x2 image pattern classification. With carefully selected architecture parameters, high-quality synthetic training data, and modern stabilization and regularization techniques (like gradient clipping and early stopping), the system achieves excellent accuracy (around 97%) in a very short time. This forms a solid foundation for more advanced image pattern recognition applications.
