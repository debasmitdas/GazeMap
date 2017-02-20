# -*- coding: utf-8 -*-
"""
GazeFollow
Created on Thu Feb 16 17:44:04 2017

@author: debasmit
"""

#from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides,padding):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k, s,padding):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding)

def alexnet1st5(x, weights, biases):
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 4,'SAME');
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0;
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias);
    maxpool1=maxpool2d(lrn1, 3,2,'VALID');
    
    conv2 = conv2d(maxpool1, weights['wc2'], biases['bc2'], 1,'SAME');
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias);
    maxpool2=maxpool2d(lrn2, 3,2,'VALID');
    
    conv3 = conv2d(maxpool2, weights['wc3'], biases['bc3'], 1,'SAME');
    
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 1,'SAME');
    
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], 1,'SAME');
    
    maxpool5=maxpool2d(conv5, 3,2, 'VALID');
    
    return maxpool5


def saliency_ext(x_i, weights, biases):
    # x_i is the whole image after postprocessing
    alex_out=alexnet1st5(x_i, weights, biases);
    saliency_out=conv2d(alex_out, weights['wc6'], biases['bc6'],1,'SAME')
    
    return saliency_out
    

def gazeFollow(x_i, x_h, x_p, weights, biases):
    
    saliency_out=saliency_ext(x_i, weights, biases)
    gaze_out=gaze_ext(x_h, x_p, weights, biases)
    
    salGazeProd=tf.multiply(saliency_out, gaze_out)
    
    #Now the 5 different shifted grided output need to be decided
    salGazeProdfc = tf.reshape(salGazeProd, [-1, weights['wSG1'].get_shape().as_list()[0]])
    
    # 1st shifted grid output
    fcSG1 = tf.add(tf.matmul(salGazeProdfc, weights['wSG1']), biases['bSG1'])
    
    # 2nd shifted grid output
    fcSG2 = tf.add(tf.matmul(salGazeProdfc, weights['wSG2']), biases['bSG2'])
    
    # 3rd shifted grid output
    fcSG3 = tf.add(tf.matmul(salGazeProdfc, weights['wSG3']), biases['bSG3'])
    
    # 4th shifted grid output
    fcSG4 = tf.add(tf.matmul(salGazeProdfc, weights['wSG4']), biases['bSG4'])
    
    # 5th shifted grid output
    fcSG5 = tf.add(tf.matmul(salGazeProdfc, weights['wSG5']), biases['bSG5'])
    
    fcSG=[fcSG1, fcSG2, fcSG3, fcSG4, fcSG5]
    
    return fcSG

#def heatmap(fcSG, alpha):
    # fcSG is the input containing the output of the fullyconnected layers 
    
    
    
    
    
    
    
    

def gaze_ext(x_h, x_p, weights, biases):
    # x_h is the head image after post processing 
    # x_p is the eye postion grid after postprocessing and flattening
    alex_out=alexnet1st5(x_h, weights, biases);
    # Here g stands for gaze
    fc6g = tf.reshape(alex_out, [-1, weights['wf6g'].get_shape().as_list()[0]])
    fc6g = tf.add(tf.matmul(fc6g, weights['wf6g']), biases['bf6g'])
    fc6g = tf.nn.relu(fc6g)
    
    fc7in=tf.concat([fc6g,x_p],0)
    fc7g = tf.add(tf.matmul(fc7in, weights['wf7g']), biases['bf7g'])
    fc7g = tf.nn.relu(fc7g)
    
    fc8g = tf.add(tf.matmul(fc7g, weights['wf8g']), biases['bf8g'])
    fc8g = tf.nn.relu(fc8g)       
    
    fc9g = tf.add(tf.matmul(fc8g, weights['wf9g']), biases['bf9g'])
    fc9g = tf.nn.sigmoid(fc9g)
    
    #Then we reshape this into 13 times 13
    fc10g = tf.reshape(fc9g, [1,1,13,13,])
    
    #Doing a convolution
    gaze_out=tf.nn.conv2d(fc10g, weights['wcg'], strides=[1, 1, 1, 1],'SAME')
    gaze_out = tf.nn.bias_add(gaze_out, biases['wcg'])
    
    return gaze_out
    
    
    
    
    
    
    
    
  
    
    
    
    
# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables

# Uncomment this section if I want to use GPU
init = tf.global_variables_initializer()
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )


print("Training will start now!")

# Launch the graph
with tf.Session(config=config) as sess:
    with tf.device("/cpu:0"):
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
    
        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.}))
