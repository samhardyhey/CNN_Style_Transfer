import numpy as np
from keras.applications.vgg16 import VGG16
from keras import backend as K
import time
import scipy

from PIL import Image
import util

##scalar weights, can be experimented with
#content_weight = 0.025
#style_weight = 5.0
#total_variation_weight = 1.0
#height = 200
#width = 200
#
##content loss
#def content_loss(content, combination):
#    return K.sum(K.square(combination - content))
#
##style loss
#def gram_matrix(x):
#    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
#    gram = K.dot(features, K.transpose(features))
#    return gram
#
#def style_loss(style, combination):
#    S = gram_matrix(style)
#    C = gram_matrix(combination)
#    channels = 3
#    size = height * width
#    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
#
##total variation loss
#def total_variation_loss(x):
#    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
#    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
#    return K.sum(K.pow(a + b, 1.25))
#
##combine all loss functions, calculate aggregate loss
#def total_loss(model, combination_image):
#    loss = K.variable(0.)
#    
#    layers = dict([(layer.name, layer.output) for layer in model.layers])
#    layer_features = layers['block2_conv2']
#    content_image_features = layer_features[0, :, :, :]
#    combination_features = layer_features[2, :, :, :]
#    
#    #content loss
#    loss += content_weight * content_loss(content_image_features,
#                                      combination_features)
#    
#    #style loss
#    feature_layers = ['block1_conv2', 'block2_conv2',
#                  'block3_conv3', 'block4_conv3',
#                  'block5_conv3']
#    for layer_name in feature_layers:
#        layer_features = layers[layer_name]
#        style_features = layer_features[1, :, :, :]
#        combination_features = layer_features[2, :, :, :]
#        sl = style_loss(style_features, combination_features)
#        loss += (style_weight / len(feature_layers)) * sl
#        
#    #total variation loss
#    loss += total_variation_weight * total_variation_loss(combination_image)
#    return loss #return total loss
#
##optimize using L-BFGS
#def eval_loss_and_grads(x):
#    x = x.reshape((1, height, width, 3))
#    
#    outputs = [loss]
#    outputs += grads
#    f_outputs = K.function([combination_array], outputs)
#    
#    outs = f_outputs([x])
#    loss_value = outs[0]
#    grad_values = outs[1].flatten().astype('float64')
#    return loss_value, grad_values
#
#class Evaluator(object):
#
#    def __init__(self):
#        self.loss_value = None
#        self.grads_values = None
#
#    def loss(self, x):
#        assert self.loss_value is None
#        loss_value, grad_values = eval_loss_and_grads(x)
#        self.loss_value = loss_value
#        self.grad_values = grad_values
#        return self.loss_value
#
#    def grads(self, x):
#        assert self.loss_value is not None
#        grad_values = np.copy(self.grad_values)
#        self.loss_value = None
#        self.grad_values = None
#        return grad_values
#
#def minimize_loss():
#    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128
#    evaluator = Evaluator()
#    iterations = 1
#    
#    for i in range(iterations):
#        print('Start of iteration', i)
#        start_time = time.time()
#        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
#                                         fprime=evaluator.grads, maxfun=20)
#        print('Current loss value:', min_val)
#        end_time = time.time()
#        print('Iteration %d completed in %ds' % (i, end_time - start_time))
#    return x

#read in images, adjust size to allow concatenation
#content image
content_image = Image.open('./base_image.jpg')
content_image = content_image.resize((200,200), Image.ANTIALIAS)
content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)

#reference image
reference_image = Image.open('./style_image.jpg')
reference_image = reference_image.resize((200,200), Image.ANTIALIAS)
reference_array = np.asarray(reference_image, dtype='float32')
reference_array = np.expand_dims(reference_array, axis=0)



#combination image
combination_array = K.placeholder((1,200,200,3))

print(content_array.shape)
print(reference_array.shape)
print(combination_array.shape)

##concatenate the images
input_tensor = K.concatenate([content_array,
                              reference_array,
                              combination_array], axis=0)

#load the model
model = VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

#calculate combination loss
loss = total_loss(model, combination_array)

#calulate gradients of generated image
grads = K.gradients(loss, combination_array)

#run optimization
x = minimize_loss()

#reshape output array
x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

#convert back into iamge
final = Image.fromarray(x)
final.show()


