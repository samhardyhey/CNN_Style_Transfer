## 1.0 Style Transfer Using Convolutional Neural Networks (CNN) ##
A python script capable of blending style attributes from one image with the
structural content of another.

A style image and content image exist as complete images. A blank "output" image is
also created. The thrust of the script is to minimize the content difference between
the content image and the output image, whilst simulatenously minimizing the style
difference between the style image and the output image. The ultimate goal is to project
the different qualities of the two images onto the blank image, fusing the two within
the new image.

Aggregating the sub-loss functions into a larger loss function allows the problem
to be abstracted away as an optimization problem. A truncated, pre-trained VGG16 model
created by Oxfords Visual Geometry Group was used as the convolution network. Whilst
scipy's L-BFGS-B optimization algorithm was used to hone the total loss experienced
by the combination image as it was developed.

Some of the results have been included, though the quality would benefit from a more
well-endowed computer and/or some AWS credits..

## 2.0 Getting Started ##
Ensure that all required libraries are available for use within the environment. In particular, ensure that
Keras is using Tensorflow as the backend.

## 2.1 Prerequisite Libraries ##
Developed using Tensorflow 1.2.0, Keras 2.0.2, PIL 4.1.1, Numpy and Scipy All of which are freely available
for use.

## 3.0 Built With ##
Python 3.6.0 - All scripting
Spyder 3.1.2 - Scientific computing IDE
Anaconda (Continum Analytics) - Popular data science platform

## 4.0 Authors ##
Sam Hardy - Implementation

## 4.1 Acknowledgments ##
Big thanks to Siraj Raval for managing to compress and spice dense topics like Neural Networks
into reasonably short youtube clips, they're an excellent starting point.
Another big thanks to Harish Narayanan for his blog entry which formed the basis for the
entire script, an excellent overview of CNN as well as some very necessary pre-material (linear, non-linear classifiers, score functions)

## 4.2 Resources ##
| Author/Title  | Link  |
| ------------- | ----- |
| "Convolutional neural networks for artistic style transfer" - Harish Narayanan     | https://harishnarayanan.org/writing/artistic-style-transfer/ |
| "How Convolutional neural networks see the world" - Keras Blog     |   https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html |
| "MNIST for ML Beginners" - Tensorflow Blog |    https://www.tensorflow.org/get_started/mnist/beginners |
| "How to Generate Art - Intro to Deep Learning #8" - Siraj Raval | https://www.youtube.com/watch?v=Oex0eWoU7AQ |
