from keras import backend
import time
import numpy as np
import scipy
#from scipy.optimize import fmin_1_bfgs_b

#scalar weights, can be experimented with
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0
height = 612
width = 612

#content loss
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

#style loss
def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#total variation loss
def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

#combine all loss functions, calculate aggregate loss
def total_loss(model, combination_image):
    loss = backend.variable(0.)
    
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = layers['block2_conv2']
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    
    #content loss
    loss += content_weight * content_loss(content_image_features,
                                      combination_features)
    
    #style loss
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

#optimize using L-BFGS
def eval_loss_and_grads(x, combination_image, loss, grads):
    x = x.reshape((1, height, width, 3))
    
    outputs = [loss]
    outputs += grads
    f_outputs = backend.function([combination_image], outputs)
    
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x, combination_image, loss, grads):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, combination_image, loss, grads)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def minimize_loss(grads, loss, combination_image):
    
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128
    evaluator = Evaluator()

    iterations = 10
    
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = scipy.optimize.fmin_1_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))





