
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import numbers
from tensorflow import random_normal, shape
from tensorflow.python.training import optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops

class SessionWrap(object):
    
    def __init__(self, session=None):
        self.session = session
        if session is None:
            self.release_session = True
        else:
            self.release_session = False

    def __enter__(self):
        if self.session is None:
            self.session = tf.Session()
        return self.session

    def __exit__(self, *args):
        if self.release_session:
            self.session.close()

class Momentum(optimizer.Optimizer):
    def __init__(self, learning_rate=1e-4, mdecay=0.05, use_locking=False, name='Momentum'):
        super(Momentum, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._mdecay = mdecay
        self._eps_t = None
        self._p = None

    def _prepare(self):
        if isinstance(self._learning_rate, numbers.Number):
            self._eps_placeholder = None
            self._set_eps_op = None
            self._eps_t = ops.convert_to_tensor(self._learning_rate, name="epsilon")
        else:
            self._eps_placeholder = tf.placeholder(self._learning_rate.dtype, [1])
            self._set_eps_op = state_ops.assign(self._learning_rate, self._eps_placeholder)
            self._eps_t = self._learning_rate

    def _create_slots(self, var_list):
        if self._p is None:
            with ops.colocate_with(var_list[0]):
                for v in var_list:
                    self._p = self._zeros_slot(v, "p", self._name)
            
    def _optimizer_step(self, grad, var, p):
        p_t = p - self._eps_t * grad - self._mdecay * p
        var_t = var + p_t
        return var_t, p_t
            
    def _apply_dense(self, grad, var):
        p = self.get_slot(var, "p")

        var_t, p_t = self._optimizer_step(grad, var, p)
        p_update   = state_ops.assign(p, p_t, use_locking=self._use_locking)
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        all_updates = [p_update, var_update]
        return control_flow_ops.group(*all_updates)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("apply_sparse not yet implemented")

    @property
    def set_eps_op(self):
        return self._set_eps_op, self._eps_placeholder

    def unroll_step(self, grads,var_list, opt_vars=None):
        next_opt_vars = []
        next_vars = []
        for i,(grad, var) in enumerate(zip(grads, var_list)):
            if opt_vars is None:
                p = self.get_slot(var, "p")
            else:
                p = opt_vars[i]
            var_t, p_t = self._optimizer_step(grad, var, p)
            next_opt_vars.append(p_t)
            next_vars.append(var_t)
        return next_vars, next_opt_vars

    
            
class SMORMS3(optimizer.Optimizer):

    def __init__(self, learning_rate=1e-4, use_locking=False, name='SMORMS3'):
        super(SMORMS3, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._eps_t = None
        self._g2 = None
        self._g = None
        self._xi = None

    def _prepare(self):
        if isinstance(self._learning_rate, numbers.Number):
            self._eps_placeholder = None
            self._set_eps_op = None
            self._eps_t = ops.convert_to_tensor(self._learning_rate, name="epsilon")
        else:
            self._eps_placeholder = tf.placeholder(self._learning_rate.dtype, [1])
            self._set_eps_op = state_ops.assign(self._learning_rate, self._eps_placeholder)
            self._eps_t = self._learning_rate

    def _create_slots(self, var_list):
        if self._g2 is None:
            with ops.colocate_with(var_list[0]):
                for v in var_list:
                    self._xi = self._zeros_slot(v, "xi", self._name)
                    self._g = self._zeros_slot(v, "g", self._name)
                    self._g2 = self._zeros_slot(v, "g2", self._name)

    def _optimizer_step(self, grad, var, xi, g, g2):
        eps = 1e-16
        r_t =  1. / (xi + 1.)
        g_t = (1. - r_t) * g + r_t * grad
        g2_t = (1. - r_t) * g2 + r_t * grad**2
        var_t = var - grad * tf.minimum(g_t * g_t / (g2_t + eps), self._eps_t) /                 (tf.sqrt(g2_t + eps) + eps)
        xi_t = 1 + xi * (1 - g_t * g_t / (g2_t + eps))
        return var_t, xi_t, g_t, g2_t
            
    def _apply_dense(self, grad, var):
        xi = self.get_slot(var, "xi")
        g = self.get_slot(var, "g")
        g2 = self.get_slot(var, "g2")
        var_t, xi_t, g_t, g2_t = self._optimizer_step(grad, var, xi, g, g2)
        
        # update helper variables
        xi_update = state_ops.assign(xi, xi_t, use_locking=self._use_locking)
        g_update = state_ops.assign(g, g_t, use_locking=self._use_locking)
        g2_update = state_ops.assign(g2, g2_t, use_locking=self._use_locking)
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        all_updates = [xi_update, g_update, g2_update, var_update]
        return control_flow_ops.group(*all_updates)
            
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("apply_sparse not yet implemented")

    @property
    def set_eps_op(self):
        return self._set_eps_op, self._eps_placeholder

    def unroll_step(self, grads,var_list, opt_vars=None):
        next_opt_vars = []
        next_vars = []
        for i,(grad, var) in enumerate(zip(grads, var_list)):
            if opt_vars is None:
                xi = self.get_slot(var, "xi")
                g = self.get_slot(var, "g")
                g2 = self.get_slot(var, "g2")
            else:
                xi, g, g2 = opt_vars[i]
            var_t, xi_t, g_t, g2_t = self._optimizer_step(grad, var, xi, g, g2)
            next_opt_vars.append([xi_t, g_t, g2_t])
            next_vars.append(var_t)
        return next_vars, next_opt_vars
              
            
def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])

def random_batch(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    b = np.random.randint(0, batches)
    start = b * size
    end = (b + 1) * size
    if len(data) == 1:
        return data[0][start:end]
    else:
        return tuple([d[start:end] for d in data])


def dense(input, output_dim, scope=None, stddev=1., reuse=False, normalized=False, params=None):
    norm = tf.contrib.layers.variance_scaling_initializer(stddev)
    const = tf.constant_initializer(0.0)
    if params is not None:
        w, b = params
    else:
        with tf.variable_scope(scope or 'linear') as scope:
            if reuse:
                scope.reuse_variables()
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm, dtype=tf.float32)
            b = tf.get_variable('b', [output_dim], initializer=const, dtype=tf.float32)
    if normalized:
        w_n = w / tf.reduce_sum(tf.square(w), 1, keep_dims=True)
    else:
        w_n = w
    return tf.matmul(input, w_n) + b



# In[2]:


import os
from os.path import expanduser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from scipy.misc import imsave

#from utils import SMORMS3, Momentum, SessionWrap, random_batch, dense

def lrelu(x, leakiness=0.2):
    return tf.maximum(leakiness * x, x)


# In[3]:


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

#def save_images(fname, flat_img, width=28, height=28, sep=3):
    #N = flat_img.shape[0]
    #pdim = int(np.ceil(np.sqrt(N)))
    #image = np.zeros((pdim * (width+sep), pdim * (height+sep)))
    #for i in range(N):
        #row = int(i / pdim) * (height+sep)
        #col = (i % pdim) * (width+sep)
        #image[row:row+width, col:col+height] = flat_img[i].reshape(width, height)
    #imsave(fname, image)
    
    
def save_images(fname, flat_img, width=28, height=28, sep=3):
    d,display = plt.subplots(1,5)
    for i in range(5):
        display[i].imshow(flat_img[i].reshape((28,28)))
    file_name = fname
    plt.savefig(file_name)
    plt.show()
    plt.close()




# In[4]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_images = np.reshape(mnist.train.images, (-1,28,28))
dataset_size = mnist_images.shape[0]
def load_mnist():
    return mnist_images, np.ones(dataset_size, dtype = 'float32')



# In[5]:


def generator(z, n_hid=500, isize=28*28, reuse=False, use_bn=False):
    bn1 = batch_norm(name='bn1')
    bn2 = batch_norm(name='bn2')
    hid = dense(z, n_hid, scope='l1', reuse=reuse)
    if use_bn:
        hid = tf.nn.relu(bn1(hid, train=True))
    else:
        hid = tf.nn.relu(hid)
    hid = dense(hid, n_hid, scope='l2', reuse=reuse)
    if use_bn:
        hid = tf.nn.relu(bn2(hid, train=True))
    else:
        hid = tf.nn.relu(hid)
    out = tf.nn.sigmoid(dense(hid, isize, scope='g_out', reuse=reuse))
    return out

def discriminator(x, z_size, n_hid=500, isize=28*28, reuse=False):
    #bn1 = batch_norm(name='bn1')
    #bn2 = batch_norm(name='bn2')
    hid = dense(x, n_hid, scope='l1', reuse=reuse, normalized=True)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    hid = dense(hid, n_hid, scope='l2', reuse=reuse, normalized=True)
    #hid = tf.nn.dropout(hid, 0.2)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    out = dense(hid, 1, scope='d_out', reuse=reuse)
    return out

def discriminator_from_params(x, params, isize=28*28, n_hid=100):
    #bn1 = batch_norm(name='bn1')
    #bn2 = batch_norm(name='bn2')
    hid = dense(x, n_hid, scope='l1', params=params[:2], normalized=True)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    hid = dense(hid, n_hid, scope='l2', params=params[2:4], normalized=True)
    hid = tf.nn.relu(hid)
    #hid = tf.tanh(hid)
    out = dense(hid, 1, scope='d_out', params=params[4:])
    return out


# In[6]:


def train(loss_d, loss_g, opt_d, opt_g, data, feed_x, feed_z, z_gen, n_steps, batch_size,
          d_steps=1, g_steps=1, d_pretrain_steps=1,
          session=None, callbacks=[]):
    with SessionWrap(session) as sess:

        sess.run(tf.initialize_all_variables())
        for t in range(n_steps):
            for i in range(d_steps):
                x = random_batch(data, size=batch_size)
                z = z_gen()
                _,curr_loss_d = sess.run([opt_d, loss_d], feed_dict = { feed_x : x, feed_z : z})
            if t > d_pretrain_steps:
                for i in range(g_steps):
                    z = z_gen()
                    _,curr_loss_g = sess.run([opt_g, loss_g], feed_dict = { feed_z : z, feed_x : x})
            else:
                curr_loss_g = 0.    
            for callback in callbacks:
                callback(t, curr_loss_d, curr_loss_g)

g = tf.Graph()
x_data, y_data = load_mnist()

x_flat = x_data.reshape((-1, 28*28))


# In[7]:


###### NOTE #######
# set number of unrolling steps and whether to use batch norm here

# This for example does not work, this also does not work when using a different optimizer (see SMORMS3 below)
lookahead = 5
use_bn = False
# but this does work (as does any setup with use_bn True)
#lookahead = 1
#use_bn = True
g_lr = 0.005
d_lr = 0.0001

eps = 1e-6
batch_size = 128
z_size = 100
g_steps = 1
d_steps = 1
steps = 30000
d_pretrain_steps = 1
isize = 28*28
with g.as_default():
    x_tf = tf.placeholder(tf.float32, shape=(batch_size, isize))
    z_tf = tf.placeholder(tf.float32, shape=(batch_size, z_size))
    with tf.variable_scope('G') as scope:
        x_gen = generator(z_tf, use_bn=use_bn)
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        g_prior = 0.
        for param in g_params:
            g_prior += 0. * tf.reduce_sum(tf.square(param))

    with tf.variable_scope('D') as scope:
        disc_out = discriminator(tf.concat([x_tf,x_gen], 0), z_size)
        disc_real = disc_out[:batch_size, :]
        disc_fake = disc_out[batch_size:, :]
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([disc_real, disc_fake], 0),
                                                                    labels=tf.concat([tf.ones_like(disc_real),
                                                                                tf.zeros_like(disc_fake)], 0)))


    # select optimizer for d
    #optimizer_d = Momentum(learning_rate=1e-3, mdecay=0.5)
    optimizer_d = SMORMS3(learning_rate=d_lr)
    opt_d = optimizer_d.minimize(loss_d, var_list=d_params)

    # unroll optimizer for G
    opt_vars = None
    next_d_params = d_params
    if lookahead > 0:
        for i in range(lookahead):
            disc_out_g = discriminator_from_params(tf.concat([x_tf, x_gen], 0), next_d_params)
            disc_real_g = disc_out_g[:batch_size, :]
            disc_fake_g = disc_out_g[batch_size:, :]
            loss_d_tmp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([disc_real_g, disc_fake_g], 0),
                                                                                labels=tf.concat([tf.ones_like(disc_real_g),
                                                                                          tf.zeros_like(disc_fake_g)], 0)))

            grads = tf.gradients(loss_d_tmp, next_d_params)
            next_d_params, opt_vars = optimizer_d.unroll_step(grads, next_d_params, opt_vars=opt_vars)
    else:
        disc_out_g = discriminator_from_params(tf.concat([x_tf, x_gen], 0), next_d_params)
        disc_fake_g = disc_out_g[batch_size:, :]
    #disc_out_g = discriminator_from_params(x_gen, next_d_params)
    #disc_fake_g = disc_out_g[batch_size:, :]
    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g, labels=tf.ones_like(disc_fake_g)))

    
    loss_generator = loss_g
    #optimizer_g = Momentum(learning_rate=1e-2, mdecay=0.5)
    optimizer_g = SMORMS3(learning_rate=g_lr)

    opt_g = optimizer_g.minimize(loss_generator, var_list=g_params)

    session = tf.Session()

    def z_gen():
        return np.random.normal(0,1, size=(batch_size, z_size))

    z_vis = z_gen()
    def logging(t, curr_loss_d, curr_loss_g):        
        if t % 100 == 0:
            print("{} loss D = {} loss G = {}".format(t, curr_loss_d, curr_loss_g))
        if t % 500 == 0:
            save_images('samples.png', session.run(x_gen, feed_dict={z_tf : z_vis}))
    print("starting")
    train(loss_d, loss_generator, opt_d, opt_g, x_flat, x_tf, z_tf, z_gen,
          steps, batch_size,
          g_steps=g_steps, d_steps=d_steps, d_pretrain_steps=d_pretrain_steps,
          session=session,
          callbacks=[logging])
    saver = tf.train.Saver()
    save_path = saver.save(session, "./mnist_samples2/mnist.ckpt")
    print("Model saved in path: %s" % save_path)


# In[8]:



for i in range(500):
    z_vis = z_gen()
    final_samples = session.run(x_gen, feed_dict={z_tf : z_vis})
    for j in range(batch_size):
        img = final_samples[j].reshape(28,28)
        file_name = './mnist_samples2/'+str(batch_size*i+j)+'.npy'
        np.save(file_name, img)
        if j==0:
            plt.imshow(img)
            plt.show()
            plt.close()


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for i in range(100):
    d, display = plt.subplots(1,5)
    for j in range(5):
        img = mpimg.imread('./mnist_samples2/'+str(5*i+j)+'.png')
        display[j].imshow(img)
    plt.show()

