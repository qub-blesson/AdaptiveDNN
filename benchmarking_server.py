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
import random
import socketserver
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

if len(sys.argv) < 2:
    print('Run like this: sudo python3 benchmarking_server.py <network_name>')
    exit()


model_name = sys.argv[1]
model = set_model(model_name)
images = ['dog', 'cat', 'tree', 'car', 'banana']
weights = [0,0,0,0,0]
# Reset 'tc' rules, in case of previous runs. 'tc' is used to slow the network connection.
command = 'tc qdisc del dev ens5 root'
p = Popen(command, stdin=PIPE, stderr=PIPE, shell=True, universal_newlines=True)


valid_cut_points = get_valid_cut_points(model, model_name)
valid_cut_points_iter = iter(valid_cut_points)

HOST = '0.0.0.0'  # The server's hostname or IP address
PORT = 1234        # The port used by the server
cp = 0
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
            cut_point = 0

        starting_layer_name = model.layers[cut_point+1].name
        print(starting_layer_name)
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

        # create the sub-model
        modelB = keras.Model(new_input, new_output)

    except Exception as e:
        print(e)
        for layer in model.layers:
            if len(layer._inbound_nodes) > 1:
                layer._inbound_nodes.pop()
        raise Exception('e')
        return None, None

    for layer in model.layers:
        if len(layer._inbound_nodes) > 1:
            layer._inbound_nodes.pop()

    return modelA, modelB


def cut_model(model, cut_point):

    modelA, modelB = cut_model_functional(model, cut_point)

    return modelA, modelB

models = []
for c in valid_cut_points:
    try:
        modelA, modelB = cut_model(model, c)
        models.append(modelB)
    except Exception as e:
        models.append(None)
        continue

    warmup_start = time.time()

    if model_name == 'lenet':
        image = load_img('images/'+images[0]+'.jpg', color_mode='grayscale', target_size=(28, 28))
    else:
        image = load_img('images/'+images[0]+'.jpg', target_size=(224, 224))

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    image = np.array([image for x in range(1)])


    input_data = modelA.predict(image)
    try:
        temp = modelB.predict(input_data)
    except Exception as e:
        pass

    a = time.time()
    print('time spent on model(s) warmup:', a - warmup_start)
    print('done warmup prediction')

max_layer = 999

class Handler(socketserver.BaseRequestHandler):

    tmp = tempfile.NamedTemporaryFile(mode='w+b')
    previous_cut_point = 0
    requests_answered = 0

    def receive_file(self):
        data = b''
        while True:

            d = self.request.recv(10240000)
            if not d:
                break
            data += d
            if d[-3:] == b'eof':
                break
        return data

    def receive_message(self):
        raw = self.request.recv(1024)
        try:
            data = json.loads(raw.decode('utf-8'))
        except Exception as e:
            print(e, raw)
            data = json.loads('{"edge_stats": [], "cloud_edge_time": {"total": -1, "edge": -1} }')

        return data

    def send_message(self, message, s):

        data = json.dumps(message, sort_keys=False, indent=2)
        data = data.ljust(1024 - len(data), ' ')
        s.sendall(data.encode())
        return True


    def handle(self):

        global cut_point, weights, pro, valid_cut_points, valid_cut_points_iter, models

        t1, t2 = 0, 0
        t3, t4 = 0, 0
        
        addr = self.client_address

        print('Got connection from', addr[0], ':', addr[1])

        cut_point_averages = []
        edge_only_averages = []
        cloud_only_averages = []

        cut_frequency = 3

        averages = []
        edge_averages = []
        cloud_averages = []
        avgs = []
        cps = []

        q = -1
        while True:
            q += 1

            # After every X requests check if cut_point should be updated
            if self.requests_answered % cut_frequency == 0:
                print('waiting for stats')
                msg = self.receive_message()
                edge_stats = msg['edge_stats']

                if self.requests_answered != 0 and q % cut_frequency == 0:
                    averages.append(sum(cut_point_averages[1:]) / (cut_frequency - 1))
                    cut_point_averages = []

                    edge_averages.append(sum(edge_only_averages[1:]) / (cut_frequency - 1))
                    edge_only_averages = []

                    cloud_averages.append(sum(cloud_only_averages[1:]) / (cut_frequency - 1))
                    cloud_only_averages = []

                    cps.append(cut_point)
                    print('cps', cps)

                print('cut_point', cut_point)
                t3 = time.time()
                cut_point = next(valid_cut_points_iter)

                print('new cut point', cut_point)
                modelB = models[valid_cut_points.index(cut_point)]
                t4 = time.time()
                # Server sends cut_point to client
                self.send_message({'cut_point': cut_point}, self.request)


            if cut_point != -1:
                data = self.receive_file()

                if not data:
                    print(addr[0], ':', addr[1], "disconnected")
                    break

                with open(self.tmp.name, 'wb') as w:
                    w.write(data)

                data_from_client = np.load(self.tmp.name, allow_pickle=True)['arr_0']

                try:
                    print('Doing inference')
                    choice = 0
                    try:
                        a = time.time()
                        output = modelB.predict(data_from_client)
                        b = time.time()
                        # print(len(modelB.layers), '/', len(model.layers))

                        self.send_message({'prediction': 'test'}, self.request)
                        times = self.receive_message()
                        cloud_edge_time = times['cloud_edge_time']['total']
                        edge_only_time = times['cloud_edge_time']['edge']
                        if int(cloud_edge_time) == -1 or int(edge_only_time) == -1:
                            q -= 1
                            cut_point = valid_cut_points[valid_cut_points.index(cut_point) - 1]
                            self.send_message({'redo': 'yes'}, self.request)
                            print('redoing cut-point ... ('+ str(cut_point) +')')
                            continue
                        else:
                            self.send_message({'redo': 'no'}, self.request)
                        cloud_only_time = b - a
                        print('time taken:', cloud_edge_time, ', edge:', edge_only_time, ', cloud:', cloud_only_time)
    

                        cut_point_averages.append(cloud_edge_time)
                        edge_only_averages.append(edge_only_time)
                        cloud_only_averages.append(cloud_only_time)
                        gc.collect()
                        print(cut_point_averages, 'avg:', sum(cut_point_averages[1:]) / (cut_frequency - 1))
                    except ValueError as e:
                        print(e)
                        self.send_message({'prediction': 'err'}, self.request)

                    
                    self.requests_answered += 1
                except Exception as e:
                    print("error", e)
                    self.send_message({'error': e}, self.request)

            if cut_point == -1:

                q -= 1

                valid_cut_points_iter = iter(valid_cut_points)

                t1 = time.time()

                with open('benchmarking_outputs/outputs_'+str(model_name)+ '/weights_'+"".join([str(x) for x in weights])+'.txt', 'w') as f:
                    for i in range(len(averages)):
                        # if i >= 0 and cps[i] >= 0:
                        if str(averages[i]) != '0.0':
                            f.write(str(averages[i]) + ' ' + str(edge_averages[i]) + ' ' + str(cloud_averages[i]) + ' ' + str(cps[i]) + '\n')

                averages = []
                edge_averages = []
                cloud_averages = []
                cps = []
    
                cut_point = 0
                try:
                    for process in psutil.process_iter():
                        if 'stress-ng' in process.name():
                            process.kill()
                except:
                    pass

                add(weights, [0,0,0,0,1], 3)

                print('new weights', weights)

                if weights[0] == 0:
                    network_speed = 50
                elif weights[0] == 1:
                    network_speed = 25
                else:
                    network_speed = 10

                if weights[1] == 0:
                    server_cpu_stress = 0
                elif weights[1] == 1:
                    server_cpu_stress = 50
                else:
                    server_cpu_stress = 100

                if weights[2] == 0:
                    memory_stress = 0
                elif weights[2] == 1:
                    memory_stress = 50
                else:
                    memory_stress = 100

                if weights[3] == 0:
                    server_memory_stress = 0
                elif weights[3] == 1:
                    server_memory_stress = 50
                else:
                    server_memory_stress = 100

                if weights[4] == 0:
                    cpu_stress = 0
                elif weights[4] == 1:
                    cpu_stress = 50
                else:
                    cpu_stress = 100

                try:
                    # Try to update stress levels, if it fails because of too much memory usage run the garbage collector and try again.
                    set_cpu_memory_stress(server_cpu_stress, server_memory_stress)
                except MemoryError:
                    gc.collect()
                    set_cpu_memory_stress(server_cpu_stress, server_memory_stress)

                t2 = time.time()


socketserver.TCPServer.allow_reuse_address = True
class ThreadedTCPServer(socketserver.TCPServer): pass

with ThreadedTCPServer((HOST, PORT), Handler) as server:
    print('Server running and waiting for connections ...')
    server.serve_forever()