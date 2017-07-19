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
from os import listdir, mkdir, getcwd

#keras backend
from keras.applications.vgg16 import VGG16 #"OxfordNet", pretrained network
from keras import backend as K

#-----------------------------------------------------------------------------

#constants, functions and classes
#1.0 model weights and constants
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

#low image dimensions because of hardware constraints
height = 50
width = 50

#2.0 loss functions 
def content_loss(content, combination):
    '''
    Calculate and return the content loss between the content image and the
    combination image. The scaled Euclidean distance between feature
    representations of the content and combination images.
    
    @content: np array representing the content image
        
    @combination: np array representing the combination image
    
    '''
    return K.sum(K.square(combination - content))

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

def total_variation_loss(x):
    '''
    Reduce the noise present within the combined image by encnouraging spatial
    smoothness via regularization.
    
    @x: the combination image.
    
    '''
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

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
    
    #style loss calculation, retrieve all layers available for manipulation within VGG model
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

def minimize_loss(save_dir, combination):
    '''
    Using stochastic gradient descent, via fmin_l_bfgs_b algorithm. Minimize
    and balance the loss experienced over the course of 10 iterations.
    Save the intermediate image combinations.
    
    '''
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128
    evaluator = Evaluator()
    iterations = 2
    
    print("\n\nProcessing: " + combination)
    for i in range(iterations):
        
        #diagnostic information
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

        #convert and save intermediate images, return the final image
        intermediate = convert_to_image(x)
        file_name = save_dir + '/' + combination + str(i) + ".jpeg" #save intermediate images
        intermediate.save(file_name, "jpeg")
    return x

#3.0 Evaluator class
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

#4.0 util functions
def convert_to_image(img_array):
    '''
    Reshape a given np array, convert back into image and return image.
    
    @img_array: a np array to be converted to an image
    
    '''
    img_array = img_array.reshape((height, width, 3)) #reshape
    img_array = img_array[:, :, ::-1]
    img_array[:, :, 0] += 103.939
    img_array[:, :, 1] += 116.779
    img_array[:, :, 2] += 123.68
    img_array = np.clip(img_array, 0, 255).astype('uint8')
    
    return Image.fromarray(img_array) #return image

#resize, convert to np array, add placeholder dimension, subtract mean RGB values
def prelim_img_process(image):
    '''
    Resize, convert to np array, add placeholder dimension, subtract the
    mean RGB values of the ImageNet training set, flip RGB pixel ordering. 
    
    '''
    image = image.resize((height,width), Image.ANTIALIAS) #resize
    image = np.asarray(image, dtype='float32') #cast to np array
    image = np.expand_dims(image,axis=0) #add placeholder dimension
    
    image[:, :, :, 0] -= 103.939 #RGB values obtained from ImageNet
    image[:, :, :, 1] -= 116.779
    image[:, :, :, 2] -= 123.68
    image = image[:, :, :, ::-1]
    return image

#retrieve all image names, images, preprocess all images
def retrieve_img():
    all_content_names = listdir("contentImages")
    all_style_names = listdir("styleImages")
    
    all_content_images = []
    all_style_images = []
    
    for content in all_content_names:
        image = Image.open('./contentImages/' + str(content)) #retrieve the content image
        image = prelim_img_process(image)
        all_content_images.append(image)
        
    for style in all_style_names:
        image = Image.open('./styleImages/' + str(style)) #retrieve the style image
        image = prelim_img_process(image)
        all_style_images.append(image)
        
    return all_content_names,all_style_names,all_content_images,all_style_images

#-----------------------------------------------------------------------------

if __name__ == "__main__":

    #1.0 read in images as a np arrays - adjust size to allow for reasonable processing time
    content_names,style_names,content_img,style_img = retrieve_img()

    #2.1 process each content image
    for content_name,content_array in zip(content_names, content_img):
        mkdir('./outputImages/' + content_name[:-4]) #create intermediate directory to store content variations
        
        #2.2 create style combinations for all style images
        for style_name,style_array in zip(style_names, style_img):
            
            #3.1 store intermediate images within combination directory
            combination = content_name[:-4] + style_name[:-4]
            mkdir('./outputImages/' + content_name[:-4] + '/' + combination)
            save_dir = getcwd() + '/outputImages/' + content_name[:-4] + '/' + combination
            
            #3.2 create placeholder image, used to store merger image
            combination_array = K.placeholder((1,height,width,3))
            
            #3.3 concatenate the image arrays
            input_tensor = K.concatenate([content_array,
                                          style_array,
                                          combination_array], axis=0)
            
            #4.0 load model, iteratively merge the two images
            #4.1 load the model
            model = VGG16(input_tensor=input_tensor,
                                weights='imagenet', include_top=False)
            
            #4.2 calculate combination loss
            loss = total_loss(model, combination_array)
            
            #4.3 calulate gradients of generated image
            grads = K.gradients(loss, combination_array)
            
            #4.4 run optimization using previously calculated loss values
            x = minimize_loss(save_dir, combination)
            
            #5.0 convert and finalize np array
            final = convert_to_image(x)
            
            #5.1 save appropriately
            final.save(save_dir + 'Final.jpeg', "jpeg")
