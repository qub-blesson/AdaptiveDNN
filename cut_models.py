import numpy as np
import time
import tensorflow as tf

# Import networks
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.resnet50 as resnet50
import keras.applications.resnet_v2 as resnet50v2
import keras.applications.mobilenet as mobilenet
import keras.applications.densenet as densenet
import non_keras_networks.alexnet as alexnet
import non_keras_networks.lenet as lenet

# Import keras features
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from keras.models import Model
from keras import models
from keras import layers

# Networking features
import socket
import tempfile
import os

import gc
import signal
import subprocess
from subprocess import Popen, PIPE
import sys
import json
import psutil

# Import common functions, shared between the Cloud and the Edge
from benchmarking_common_functions import *

if len(sys.argv) < 3:
    print('Run like this: sudo python3 cut_models.py <model_name> <batch_size>')
    exit()

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
model = set_model(model_name)

images = ['dog', 'cat', 'tree', 'car', 'banana']

valid_cut_points = []
cut_point = 0

# this is the split point, i.e. the starting layer in our sub-model
starting_layer_name = model.layers[cut_point].name

# create a new input layer for our sub-model we want to construct
layer_input_shape = model.get_layer(starting_layer_name).get_input_shape_at(0)
if isinstance(layer_input_shape, list):
    layer_input = [layers.Input(shape=layer_input_shape[0][1:]) for x in range(len(layer_input_shape))]
else:
    layer_input = layers.Input(shape=layer_input_shape[1:])

new_input = layer_input
layer_outputs = {}

def get_output_of_layer(layer):
    # if we have already applied this layer on its input(s) tensors,
    # just return its already computed output
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    # if this is the starting layer, then apply it on the input tensor
    if layer.name == starting_layer_name:
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    # find all the connected layers which this layer
    # consumes their output
    prev_layers = []
    for node in layer._inbound_nodes:
        prev_layers.extend(node.inbound_layers)

    # get the output of connected layers
    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl)])

    # apply this layer on the collected outputs
    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out

# Functional models
def cut_model_functional(model, cut_point):

    global layer_outputs, starting_layer_name, new_input, valid_cut_points
    try:
        if cut_point >= len(model.layers):
            cut_point = len(model.layers)-1
        elif cut_point <= 0:
            cut_point = 1

        starting_layer_name = model.layers[cut_point+1].name
        # create a new input layer for our sub-model we want to construct
        new_input = layers.Input(batch_shape=model.get_layer(starting_layer_name).get_input_shape_at(0))
        layer_input_shape = model.get_layer(starting_layer_name).get_input_shape_at(0)
        if isinstance(layer_input_shape, list):
            layer_input = [layers.Input(shape=layer_input_shape[0][1:]) for x in range(len(layer_input_shape))]
        else:
            layer_input = layers.Input(shape=layer_input_shape[1:])
        new_input = layer_input
        layer_outputs = {}

        in_l = model.get_layer(index=0)
        out_l = model.get_layer(index=cut_point)
        modelA = Model(inputs=in_l.get_input_at(0), outputs=out_l.get_output_at(0))

        new_output = get_output_of_layer(model.layers[-1])

        valid_cut_points.append(cut_point)

        # create the sub-model
        modelB = keras.Model(new_input, new_output)

    except Exception as e:
        for layer in model.layers:
            if len(layer._inbound_nodes) > 1:
                layer._inbound_nodes.pop()
        raise Exception('dfdjf')
        return None, None

    for layer in model.layers:
        if len(layer._inbound_nodes) > 1:
            layer._inbound_nodes.pop()

    return modelA, modelB

def cut_model(model, cut_point):

    modelA, modelB = cut_model_functional(model, cut_point)

    return modelA, modelB

# Cycle through all the layers, trying split the model after each one.
for c in range(0, len(model.layers)):
    try:
        
        modelA, modelB = cut_model(model, c)

        warmup_start = time.time()
                                    
        image = load_img('images/'+images[0]+'.jpg', target_size=(224, 224))

        # convert the image pixels to a numpy array
        image = img_to_array(image)

        # image = np.expand_dims(image, axis=0)
        image = np.array([image for x in range(batch_size)])

        image = vgg16.preprocess_input(image)
        input_data = modelA.predict(image)
        test = modelB.predict(input_data)
        a = time.time()
        print('time spent on model(s) warmup:', a - warmup_start)
        print('done warmup prediction')

        model_json = modelA.to_json()
        with open('edge_models_'+str(batch_size)+'b/edge_models_'+model_name+'/model_'+ str(c) +'.json', 'w') as json_file:
            json_file.write(model_json)
        print('done model ', c)

    except Exception as e:
        print('couldn\'t cut ', c)
        print('exception', e)
        print('----------')
        pass