import numpy as np

path = '/home/neon/883Project/UnrolledGAN/mnist_samples2/'
a = np.array([np.load(path+str(i)+'.npy') for i in range(60000)])
np.save('mnist_synth_x.npy', a)

