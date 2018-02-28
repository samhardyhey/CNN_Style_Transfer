import numpy as np
import time
import scipy

from keras.applications.vgg16 import VGG16  # "OxfordNet", pretrained network
from keras import backend as K

from evaluator import Evaluator
from util import export_image


def content_loss(content_np, composition_np):
    '''
    Calculate and return the content loss between the content image and the
    combination image (Scaled Euclidean distance).
    '''

    return K.sum(K.square(composition_np - content_np))


def gram_matrix(image_np):
    '''
    Captures aggregate information about a particular image whilst ignoring 
    internal, structural detail.
    '''

    features = K.batch_flatten(K.permute_dimensions(image_np, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))

    return gram


def style_loss(style_np, composition_np, height, width):
    '''
    Calculate style loss between style reference image and combination image.
    Calculated as the scaled Frobenius norm of the difference between the Gram
    matrices of the style/combination images.
    '''

    S = gram_matrix(style_np)
    C = gram_matrix(composition_np)
    channels = 3
    size = height * width

    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(image_np, height, width):
    '''
    Reduce the noise present within the combined image by encnouraging spatial
    smoothness, achieved via regularization.
    '''

    a = K.square(image_np[:, :height - 1, :width - 1,
                          :] - image_np[:, 1:, :width - 1, :])
    b = K.square(image_np[:, :height - 1, :width - 1, :] -
                 image_np[:, :height - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))


def total_loss(model, composition_np, content_weight, style_weight,
               height, width, total_variation_weight):
    '''
    Return the total loss present within the combination image.
    Total loss is calculated by considering the content, style and variation
    loss.
    '''

    loss = K.variable(0.)

    layers = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = layers['block2_conv2']
    content_features = layer_features[0, :, :, :]
    composition_features = layer_features[2, :, :, :]

    # content loss
    loss += content_weight * \
        content_loss(content_features, composition_features)

    # style loss calculation, retrieve all layers available for manipulation
    feature_layers = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']

    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features, height, width)
        loss += (style_weight / len(feature_layers)) * sl

    # total variation loss
    loss += total_variation_weight * \
        total_variation_loss(composition_np, height, width)
    return loss  # return total loss


def minimize_loss(save_name, loss, grads, composition_np, iterations, height, width):
    '''
    Using stochastic gradient descent via fmin_l_bfgs_b algorithm. Minimize
    and balance the loss experienced over the course of 10 iterations.
    Save the intermediate image combinations.
    '''

    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128
    evaluator = Evaluator(loss, grads, composition_np, height, width)

    print("\n\nProcessing: " + save_name)
    for i in range(iterations):

        # print diagnostic information
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss,
                                                        x.flatten(),
                                                        fprime=evaluator.grads,
                                                        maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    return x


def create_composition_single(style_np, content_np, save_name, save_dir,
                              height, width, content_weight, style_weight,
                              total_variation_weight, iterations):
    '''
    Combine all components, create composite image with supplied
    content and style image arrays.
    '''

    # placeholder composition image
    composition_np = K.placeholder((1, height, width, 3))

    # concat image arrays as single tensor
    input_tensor = K.concatenate([content_np,
                                  style_np,
                                  composition_np], axis=0)

    # load model
    model = VGG16(input_tensor=input_tensor,
                  weights='imagenet', include_top=False)

    # define initial loss
    loss = total_loss(model, composition_np, content_weight,
                      style_weight, height, width,
                      total_variation_weight)

    # define initial gradients
    grads = K.gradients(loss, composition_np)

    # run optimization using previously calculated loss values
    composition = minimize_loss(save_name, loss, grads,
                                composition_np, iterations,
                                height, width)

    # 5.0 convert and finalize np array
    export_image(composition, save_dir, height, width)
