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
    print('Run like this: sudo python3 benchmarking_client.py <server_address> <network_name>')
    exit()

model_name = sys.argv[2]
model = set_model(model_name)
images = ['dog', 'cat', 'tree', 'car', 'banana']

# A temporary in-memory file object is used because it provides the same easy to use API as a regular file but it
# remains in main-memory and much faster than writing to the disk.
tmp = tempfile.NamedTemporaryFile(mode='w+b')

# Host is passed from the command-line
HOST = sys.argv[1]
# The port used by the server
PORT = 1234
model_name = sys.argv[2]

cut_point = -1
sent_requests = 0

# Set the network speed to maximum, in case it's already limited due to previous runs.
set_network_speed(50)

# Initialise variables
cut_points = []
network_speed = 50
weights = [0,0,0,0,0]
models = []
valid_cut_points = get_valid_cut_points(model, model_name)
valid_cut_points_iter = iter(valid_cut_points)

# Load the pre-partitioned Edge models into the 'models' list
for c in valid_cut_points:

    if c == -1:
        models.append(None)
        break

    json_file = open('edge_models/edge_models_'+str(model_name)+'/model_'+ str(c) +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    modelA = model_from_json(loaded_model_json)
    models.append(modelA)

    print('model layers:', len(modelA.layers))

    time.sleep(1)
    print('done sleep')

    warmup_start = time.time()
    
    # LeNet requires a different shaped input to the other networks and its images must be grayscale, not RGB
    if model_name == 'lenet':
        image = load_img('images/'+images[0]+'.jpg', color_mode='grayscale', target_size=(28, 28))
    else:
        image = load_img('images/'+images[0]+'.jpg', target_size=(224, 224))

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    image = np.array([image for x in range(1)])

    # prepare the image for the VGG model
    input_data = modelA.predict(image)
    a = time.time()
    print('time spent on model(s) warmup:', a - warmup_start)
    print('done warmup prediction')

pro = subprocess.Popen('ls', stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        s.connect((HOST, PORT))
        print("Connected to server.")
        cut_frequency = 3

        q = -1
        while True:
            q += 1
            if sent_requests % cut_frequency == 0:

                send_message({'edge_stats': {
                    'network': network_speed,
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory()[2],
                    } }, s)

                new_cut_point = receive_message(s)['cut_point']

                if new_cut_point != cut_point:
                    cut_point = new_cut_point
                    print('Server set cut point to:', cut_point)
                    modelA = models[valid_cut_points.index(cut_point)]

            if cut_point != -1:

                img_choice = 0
                
                # LeNet expects different input image dimensions
                if model_name == 'lenet':
                    image = load_img('images/'+images[0]+'.jpg', color_mode='grayscale', target_size=(28, 28))
                else:
                    image = load_img('images/'+images[0]+'.jpg', target_size=(224, 224))

                # convert the image pixels to a numpy array
                image = img_to_array(image)

                # image = np.expand_dims(image, axis=0)
                image = np.array([image for x in range(1)])

                # prepare the image for the VGG model
                a = time.time()
        
                if modelA is not None:
                    input_data = modelA.predict(image)
                else:

                    input_data = []

                x1 = time.time()
                predict_time = x1 - a
                np.savez(tmp.name, input_data, allow_pickle=True)

                x5 = time.time()

                save_data_time = x5 - x1

                x2 = time.time()
                # Send output of model from the Edge to the Cloud
                s.sendfile(open(tmp.name+'.npz', "rb"))
                s.sendall(b'eof')

                x3 = time.time()
                send_file_time = x3 - x2

                x6 = time.time()
                send_done_time = x6-x3

                answer = receive_message(s)['prediction']
                x7 = time.time()
                recv_data_time = x7-x6

                b = time.time()
                total_cloud_edge = recv_data_time + send_done_time + send_file_time + save_data_time + predict_time
                # Send the timing data back to the server
                print('time taken:', b-a)
                send_message({'cloud_edge_time': {
                    'total': b-a,
                    'edge': predict_time,
                    }}, s)
                    
                # Redo the previous measurement if the server says so.
                redo = receive_message(s)['redo']
                if redo == 'yes':
                    q -= 1
                    continue
            

            # Update the weights after every cut point has been measured
            if sent_requests % cut_frequency == 0 and cut_point == -1:

                q -= 1

                valid_cut_points_iter = iter(valid_cut_points)

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

                # Kill the stress program so it can run again for the next iteration
                try:
                    for process in psutil.process_iter():
                        if 'stress-ng' in process.name():
                            process.kill()
                except:
                    pass
                
                # Update 'tc' network speed
                set_network_speed(network_speed)

                # Try to run the stress program, if it fails, try freeing some memory and try again.
                try:
                    set_cpu_memory_stress(cpu_stress, memory_stress)
                except MemoryError:
                    gc.collect()
                    set_cpu_memory_stress(cpu_stress, memory_stress)

                # This 1 second pause prevents crashes
                print('giving CPU 1 second rest ...')
                time.sleep(1)

            sent_requests += 1
                

except Exception as e:
    print(e)
    sys.exit(0)