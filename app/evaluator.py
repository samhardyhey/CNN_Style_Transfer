import numpy as np
from keras import backend as K


def eval_loss_and_grads(x, loss, grads, composition_np, height, width):
    '''
    Define gradients of the total loss relative to the composition image, in
    order to minimize the loss.
    '''
    x = x.reshape((1, height, width, 3))
    outputs = [loss]
    outputs += grads
    f_outputs = K.function([composition_np], outputs)

    outs = f_outputs([x])
    prog_loss = outs[0]
    grad_values = outs[1].flatten().astype('float64')

    return prog_loss, grad_values


class Evaluator(object):
    '''
    Computes the loss and gradient present within the composition image. Phrased
    as a class to conveniently package retrieval methods for interfacing with 
    scypy.optimize library.
    '''

    def __init__(self, il, ig, comb_np, height, width):
        self.prog_loss = None  # bundle variables as object fields.. eek
        self.prog_grads = None
        self.initial_loss = il
        self.initial_grad = ig
        self.comb_np = comb_np
        self.height = height
        self.width = width

    def loss(self, x):
        '''
        Retrieve the loss value for the composition image.
        '''
        assert self.prog_loss is None
        self.prog_loss, self.grad_values = eval_loss_and_grads(x, self.initial_loss,
                                                               self.initial_grad,
                                                               self.comb_np,
                                                               self.height,
                                                               self.width)

        return self.prog_loss

    def grads(self, x):
        '''
        Retrieve the gradients associated with a particular composition
        image.
        '''
        assert self.prog_loss is not None
        grad_values = np.copy(self.grad_values)
        self.prog_loss = None
        self.grad_values = None

        return grad_values
