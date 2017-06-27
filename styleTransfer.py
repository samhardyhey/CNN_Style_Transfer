'''
Image style transfer using convolutional neural networks.

Based upon;

- Harish Narayanan's Blog, "Convolutional neural networks for
artistic style transfer
- https://harishnarayanan.org/writing/artistic-style-transfer/

- Keras Blog, "How Convolutional neural networks see the world"
- https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

- TensorFlow Blog, "MNIST for ML Beginners"
- https://www.tensorflow.org/get_started/mnist/beginners

- Siraj Raval, "How to Generate Art - Intro to Deep Learning #8"
- https://www.youtube.com/watch?v=Oex0eWoU7AQ

A style image, content image and output image exist. The thrust of the script is
to minimize the content difference between the content image and the output image,
whilst simulatenously minimizing the style difference between the style image
and the output image.
'''

#util
import numpy as np
import time
import scipy
from PIL import Image

#keras backend
from keras.applications.vgg16 import VGG16 #"OxfordNet", pretrained network
from keras import backend as K

#-----------------------------------------------------------------------------

#1.0 constants, functions and classes
#1.1 scalar weights, can be experimented with to change effect
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0
height = 200
width = 200

#1.2 loss functions 
#content loss
def content_loss(content, combination):
    '''
    Calculate and return the content loss between the content image and the
    combination image. The scaled Euclidean distance between feature
    representations of the content and combination images.
    
    @content: np array representing the content image
        
    @combination: np array representing the combination image
    
    '''
    return K.sum(K.square(combination - content))

#style loss functions 
def gram_matrix(x):
    '''
    Captures information about which features within an image tend to 
    activate with one another. Captures aggregate information about a particular
    image whilst ignoring internal, structural detail.
    
    '''
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    '''
    Calculate style loss between style reference image and combination image.
    Calculated as the scaled Frobenius norm of the difference between the Gram
    matrices of the style/combination images.
    
    @style: np array representing the style reference image
    
    @combination: np array representing the combination image
    
    '''
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#total variation loss
def total_variation_loss(x):
    '''
    Reduce the noise present within the combined image by cnouraging spatial
    smoothness via regularization.
    
    @x: the combination image.
    
    '''
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#combine all loss functions, calculate aggregate loss
def total_loss(model, combination_image):
    '''
    Return the total loss present within the combination image.
    Total loss is calculated by considering the content, style and variation
    loss.
    
    @model: Tensorflow CNN model, VGG16 in this case
        
    @combination_image: np array representing the combination image
    '''
    loss = K.variable(0.)
    
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = layers['block2_conv2']
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    
    #content loss
    loss += content_weight * content_loss(content_image_features,
                                      combination_features)
    
    #style loss calculation
    #retrieve all layers available for manipulation within VGG model
    feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']
    
    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
        
    #total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss #return total loss

#optimize loss using L-BFGS algorithm
def eval_loss_and_grads(x):
    '''
    Define gradients of the total loss relative to the combination image, in
    order to minimize the loss.
    
    @x: the combination image.
    '''
    
    x = x.reshape((1, height, width, 3))
    
    outputs = [loss]
    outputs += grads
    f_outputs = K.function([combination_array], outputs)
    
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

def minimize_loss():
    '''
    Using stochastic gradient descent, via fmin_l_bfgs_b algorithm. Minimize
    and balance the loss experienced over the course of 10 iterations. 
    
    '''
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128
    evaluator = Evaluator()
    iterations = 10
    
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    return x

#1.3 Evaluator class
class Evaluator(object):
    '''
    Computes the loss and gradient present within the combination image. Phrased
    as a class to package these retrievals as methods, to ensure efficiency when
    interfacing with the scypy.optimize library.
    
    '''
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        '''
        Retrieve the loss value for the combination image.
        
        @x: the combination image.
        '''
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        '''
        Retrieve the gradients associated with a particular combination
        image.
        
        @x: the combination image.
        '''
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

#-----------------------------------------------------------------------------

if __name__ == "__main__":

    #1.0 read in images as a np arrays - adjust size to allow concatenation and
    # reasonable processing time
    #1.1 content image
    content_image = Image.open('./contentImages/goldenCity.jpg')
    content_image = content_image.resize((200,200), Image.ANTIALIAS) #resize for processing restraints
    content_array = np.asarray(content_image, dtype='float32')
    content_array = np.expand_dims(content_array, axis=0) #add additional axis
    
    #1.2 reference image
    reference_image = Image.open('./styleImages/bubblePainting.jpg')
    reference_image = reference_image.resize((200,200), Image.ANTIALIAS)
    reference_array = np.asarray(reference_image, dtype='float32')
    reference_array = np.expand_dims(reference_array, axis=0)
    
    #1.3 subtract mean RGB values (as contained within ImageNet training set),
    # reverse element orderings within FGB np arrays
    content_array[:, :, :, 0] -= 103.939 #RGB values obtained from ImageNet
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68
    content_array = content_array[:, :, :, ::-1]
    
    reference_array[:, :, :, 0] -= 103.939
    reference_array[:, :, :, 1] -= 116.779
    reference_array[:, :, :, 2] -= 123.68
    reference_array = reference_array[:, :, :, ::-1]
    
    #1.4 combination image - placeholder, appropriately dimensioned
    combination_array = K.placeholder((1,200,200,3))
    
    #1.5 concatenate the images
    input_tensor = K.concatenate([content_array,
                                  reference_array,
                                  combination_array], axis=0)
    
    #2.0 load model and calculate loss types
    #2.1 load the model
    model = VGG16(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    
    #2.2 calculate combination loss
    loss = total_loss(model, combination_array)
    
    #2.3 calulate gradients of generated image
    grads = K.gradients(loss, combination_array)
    
    #3.0 run optimization using previously calculated loss values
    x = minimize_loss()
    
    #4.0 finalize and save output
    #4.1 reshape output array
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    
    #4.2 convert back into iamge, display and save image
    final = Image.fromarray(x)
    final.show()
    final.save("./outputImages/image1.jpeg", "jpeg")