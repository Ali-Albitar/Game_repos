"""conv.py
~~~~~~~~~~

Code for many of the experiments involving convolutional networks in
Chapter 6 of the book 'Neural Networks and Deep Learning', by Michael
Nielsen.  The code essentially duplicates (and parallels) what is in
the text, so this is simply a convenience, and has not been commented
in detail.  Consult the original text for more details.

"""

from collections import Counter

import matplotlib  # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import theano # type: ignore
import theano.tensor as T # type: ignore

import network3
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

def shallow(n=3, epochs=60):
    nets = []
    for j in range(n):
        print 
        "A shallow net with 100 hidden neurons, in this case we just get the  neurons from that classes to help us lee that bugs we talked in that netwppork clases "
        net = Network([
            FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, 
            validation_data, test_data)
        nets.append(net)
    return nets 

    """
   this is the main and conv class this is the most important class in the all hole project 
   the first element has multiple choises the first one is 500 and it is so important to see all the project how it worke it 
   
   we know in the all the projects we should take a breth 
    """


def basic_conv(n=3, epochs=60):
    for j in range(n + epochs ):
        print
        "conv + FC architecture = SoftmaxLayer(values of that configre data to keep it inside that machine )"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size,2 ,28 , 30),
            filter_shape=(30,3,6,6),
            poolsize=(3,3)),
            FullyConnectedLayer(n_in=20*12*12, mmini_batch_size = 100),
            SoftmaxLayer(n_in=27*13*13, n_out=10)], mini_batch_size)
        net.SGD(training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
    return net 

def Get_Images_Inveiroments(value=7 ,gradients = 75):
                """
                the first thing u should do it when u open the project is start from the begin and the most important thing that is when the procet begin everythin will be ready all the project 
                in otherwise we should take care about the client how he is feeling about the project and if he accept the feuture editing this mace eveything look very good for him. 
                """

    for i in range(value):
        print 
        'this class output is the all value we get it from the images images and the best way to do taht is this tradintional way '
        inviroments_from_old_images = Network([
            ConvPoolLayer(image_shape=(mini_batch_size,2 ,28 , 30),
            filter_shape=(30,3,6,6),
            poolsize=(3,3)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=27*13*13, n_out=10)], mini_batch_size)
        inviroments_from_old_images.SGD(training_data, gradients, mini_batch_size, 0.1, validation_data, test_data)
    return inviroments_from_old_images
        
def omit_FC():
    for j in range(3):
        print 
        " The best way to get the links from images is uplouds images from costumer folders am+nd step by step we get all inviroments the most importsnt thing in this awy is the takes the all inviroments from the main class and pot it in another place to mace the hole thing brfore the client coming to try and use the project this maybe make a seance to see how every thing work from the bahind of the all this project u should be an senior AI enginir u should have a lot of the important and at all u have this operation before the main problem   "
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            SoftmaxLayer(n_in=20*12*12, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    return net 

#this note is very importants for run this project because when that run that should stop in this case 
#when that stoped in her we should put any inviroments we get from the new images 
#in the end we should egualurizer
#egualurizer for all inviroments we get it from images v alue when we recalled in the onother class 

def dbl_conv(activation_fn=sigmoid):
    for j in range(3):
        print 
        "Conv + Conv + FC architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=100, activation_fn=activation_fn),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    return net 

# in the first we should run the previouse classes cause we need allmost 100 parameters to fix it in the same class that we called in ,
# the main problem that is how we get the range of that parameters :
# firstly we should call the parameters from the figure we get it from images.
# that using convolutional-pooling layers is already a pretty strong
# regularizer.
def regularized_dbl_conv():
    for lmbda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print 
            "Conv + Conv + FC num %s, with regularization %s" % (j, lmbda)
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2)),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2)),
                FullyConnectedLayer(n_in=40*4*4, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data, lmbda=lmbda)
#it is the  previouse classes notes cause of this class depends for all previouse classes and tha
def dbl_conv_relu():
    for lmbda in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print 
            "Conv + Conv + FC num %s, relu, with regularization %s" % (j, lmbda)
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2), 
                              activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2), 
                              activation_fn=ReLU),
                FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=lmbda)
            
            
            """
                    this may make sometimes some problems when the code is run and the reason to make it is posible that found the bug and rerinning the project 
                    the second thing is the last value that we get it from the ,main opject that we found it in the first time that we did id to reused in feurtre 
                    the last and the most important thing it is make all values in tha main classes and find the real problem and resolved several times and all vakues should be orgnised whis real values that we found 
            """
            

#### Some subsequent functions may make use of the expanded MNIST
#### data.  That can be generated by running expand_mnist.py.
#### some data should be changing in the code when u run it .
#### first thing u want doing that is enter tha images values u get it from equalizer .

def expanded_data(n=100):
    """n is the number of neurons in the fully-connected layer.  We'll try
    n=100, 300, and 1000.
    
    all this Parameters is already taken by zhe real coder that he maded this fucking code 
    to be honest this is the most bad code i have ever see it in this world firstly u need to get all values from the Backend and frontensd that he made it in this fucking real code 

    """
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print
        "Training with expanded data, %s neurons in the FC layer, run num %s" % (n, j)
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=n, activation_fn=ReLU),
            SoftmaxLayer(n_in=n, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=0.1)
    return net 
# the parameters is genareted when equalizer from the run projects 
def expanded_data_double_fc(n=100):
    """n is the number of neurons in both fully-connected layers.  We'll
    try n=100, 300, and 1000.

    """
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print 
        "Training with expanded data, %s neurons in two FC layers, run num %s" % (n, j)
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=n, activation_fn=ReLU),
            FullyConnectedLayer(n_in=n, n_out=n, activation_fn=ReLU),
            SoftmaxLayer(n_in=n, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=0.1)

def double_fc_dropout(p0, p1, p2, repetitions):
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    nets = []
    for j in range(repetitions):
        print 
        "\n\nTraining using a dropout network with parameters ",p0,p1,p2
        print 
        "Training with expanded data, run num %s" % j
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=p0),
            FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=p1),
            SoftmaxLayer(n_in=1000, n_out=10, p_dropout=p2)], mini_batch_size)
        net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, 
                validation_data, test_data)
        nets.append(net)
    return nets

def ensemble(nets): 
    """Takes as input a list of nets, and then computes the accuracy on
    the test data when classifications are computed by taking a vote
    amongst the nets.  Returns a tuple containing a list of indices
    for test data which is erroneously classified, and a list of the
    corresponding erroneous predictions.

    Note that this is a quick-and-dirty kluge: it'd be more reusable
    (and faster) to define a Theano function taking the vote.  But
    this works.

    """
    
    test_x, test_y = test_data
    for net in nets:
        i = T.lscalar() # mini-batch index
        net.test_mb_predictions = theano.function(
            [i], net.layers[-1].y_out,
            givens={
                net.x: 
                test_x[i*net.mini_batch_size: (i+1)*net.mini_batch_size]
            })
        net.test_predictions = list(np.concatenate(
            [net.test_mb_predictions(i) for i in range(1000)]))
    all_test_predictions = zip(*[net.test_predictions for net in nets])
    
    
    
    '''
    this code maded by ali to recognise how the human thinking about the real machine 
    behinde of the rel world this makes me go to another machines and that make s the other people go to dificult things to solve most of that problems 
    in this case i maded this code to resolve all bugs and problems that happend with the other coder in this live to make it more easy fr them 
    firstly u should takes the main Object from that code u will resolve it then u go to all parameters thAT INCLUDED IN THAT CODE 

    '''
    def plurality(p): return Counter(p).most_common(1)[0][0]
    plurality_test_predictions = [plurality(p) 
                                  for p in all_test_predictions]
    test_y_eval = test_y.eval()
    error_locations = [j for j in range(10000) 
                       if plurality_test_predictions[j] != test_y_eval[j]]
    erroneous_predictions = [plurality(all_test_predictions[j])
                             for j in error_locations]
    print 
    "Accuracy is {:.2%}".format((1-len(error_locations)/10000.0))
    return error_locations, erroneous_predictions

def plot_errors(error_locations, erroneous_predictions=None):
    test_x, test_y = test_data[0].eval(), test_data[1].eval()
    fig = plt.figure()
    error_images = [np.array(test_x[i]).reshape(28, -1) for i in error_locations]
    n = min(40, len(error_locations))
    for j in range(n):
        ax = plt.subplot2grid((5, 8), (j/8, j % 8))
        ax.matshow(error_images[j], cmap = matplotlib.cm.binary)
        ax.text(24, 5, test_y[error_locations[j]])
        if erroneous_predictions:
            ax.text(24, 24, erroneous_predictions[j])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
    
def plot_filters(net, layer, x, y):

    """Plot the filters for net after the (convolutional) layer number
    layer.  They are plotted in x by y format.  So, for example, if we
    have 20 filters after layer 0, then we can call show_filters(net, 0, 5, 4) to
    get a 5 by 4 plot of all filters."""
    filters = net.layers[layer].w.eval()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j)
        ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt


#### Helper method to run all experiments in the book

def run_experiments():

    """Run the experiments described in the book.  Note that the later
    experiments require access to the expanded training data, which
    can be generated by running expand_mnist.py.

    """
    shallow()
    basic_conv()
    omit_FC()
    dbl_conv(activation_fn=sigmoid)
    # omitted, but still interesting: regularized_dbl_conv()
    dbl_conv_relu()
    expanded_data(n=100)
    expanded_data(n=300)
    expanded_data(n=1000)
    expanded_data_double_fc(n=100)    
    expanded_data_double_fc(n=300)
    expanded_data_double_fc(n=1000)
    nets = double_fc_dropout(0.5, 0.5, 0.5, 5)
    
    # plot the erroneous digits in the ensemble of nets just trained
    error_locations, erroneous_predictions = ensemble(nets)
    plt = plot_errors(error_locations, erroneous_predictions)
    plt.savefig("ensemble_errors.png")
    # plot the filters learned by the first of the nets just trained
    plt = plot_filters(nets[0], 0, 5, 4)
    plt.savefig("net_full_layer_0.png")
    plt = plot_filters(nets[0], 1, 8, 5)
    plt.savefig("net_full_layer_1.png")

