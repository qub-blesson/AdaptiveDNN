# Import networks
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.resnet50 as resnet50
import keras.applications.resnet_v2 as resnet50v2
import keras.applications.mobilenet as mobilenet
import keras.applications.densenet as densenet
import non_keras_networks.alexnet as alexnet
import non_keras_networks.lenet as lenet

import signal
import subprocess
from subprocess import Popen, PIPE
import os
import json
import socket

def add(x, y, b):

    carry = 0
    for i in reversed(range(len(x))):

        if carry == 1:
            carry = 0
            y[i] += 1

        if x[i] + y[i] >= b:
            carry = 1
            x[i] = (x[i] + y[i]) % b
        else:
            x[i] += y[i]

    return x

def receive_message(s):

    data = json.loads(s.recv(1024).decode('utf-8'))

    return data

def send_message(message, s):

    data = json.dumps(message, sort_keys=False, indent=2)
    data = data.ljust(1024 - len(data), ' ')
    s.sendall(data.encode())
    return True

def set_model(model_name):

    if model_name == 'vgg16':
        return vgg16.VGG16(weights=None)
    elif model_name == 'vgg19':
        return vgg19.VGG19(weights=None)
    elif model_name == 'resnet50':
        return resnet50.ResNet50(weights=None)
    elif model_name == 'resnet50v2':
        return resnet50v2.ResNet50V2(weights=None)
    elif model_name == 'densenet':
        return densenet.DenseNet121(weights=None)
    elif model_name == 'mobilenet':
        return mobilenet.MobileNet(weights=None)
    elif model_name == 'alexnet':
        return alexnet.alexnet(weights=None)
    elif model_name == 'lenet':
        return lenet.lenet_model(weights=None)
    else:
        print('Unsupported model ...')
        return None

def get_valid_cut_points(model, model_name):
    valid_cut_points = []
    for c, val in enumerate(model.layers):
        try:
            json_file = open('edge_models/edge_models_'+str(model_name)+'/model_'+ str(c) +'.json', 'r')
            valid_cut_points.append(c)
        except IOError as e:
            pass
    print('Loaded valid cut points:', valid_cut_points)
    return valid_cut_points + [-1]

def set_cpu_memory_stress(cpu_stress, memory_stress):
    pro = subprocess.Popen('stress-ng -c 0 -l ' + str(cpu_stress), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 
    print('set cpu stress to', cpu_stress)

    pro = subprocess.Popen('stress-ng --vm-bytes $(awk \'/MemFree/{printf "'+str(memory_stress)+'";}\' < /proc/meminfo)% --vm-hang 0 -m 1', stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 
    print('set memory usage to', memory_stress)

def set_network_speed(network_speed):
    # Remove the old speed limits if there are any
    command = 'tc qdisc del dev ens5 root'
    p = Popen(command, stdin=PIPE, stderr=PIPE, shell=True, universal_newlines=True)

    # Set the new speed limit
    command = 'tc qdisc add dev ens5 root tbf rate ' + str(network_speed) + 'mbit burst 500kbit latency 50ms'
    p = Popen(command, stdin=PIPE, stderr=PIPE, shell=True, universal_newlines=True)
    print('set network speed to ', network_speed, 'mbit')