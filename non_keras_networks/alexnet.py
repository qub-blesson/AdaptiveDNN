# """
# AlexNet Keras Implementation

# BibTeX Citation:

# @inproceedings{krizhevsky2012imagenet,
#   title={Imagenet classification with deep convolutional neural networks},
#   author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
#   booktitle={Advances in neural information processing systems},
#   pages={1097--1105},
#   year={2012}
# }
# """

# # Import necessary packages
# import argparse
# import numpy as np
# # Import necessary components to build LeNet
# from keras.models import Sequential
# import keras.applications.vgg16 as vgg16
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
# from keras.regularizers import l2

# def alexnet_model(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0.,
#     weights=None):

#     # Initialize model
#     alexnet = Sequential()

#     # Layer 1
#     alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
#         padding='same', kernel_regularizer=l2(l2_reg)))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 2
#     alexnet.add(Conv2D(256, (5, 5), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 3
#     alexnet.add(ZeroPadding2D((1, 1)))
#     alexnet.add(Conv2D(512, (3, 3), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 4
#     alexnet.add(ZeroPadding2D((1, 1)))
#     alexnet.add(Conv2D(1024, (3, 3), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))

#     # Layer 5
#     alexnet.add(ZeroPadding2D((1, 1)))
#     alexnet.add(Conv2D(1024, (3, 3), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 6
#     alexnet.add(Flatten())
#     alexnet.add(Dense(3072))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(Dropout(0.5))

#     # Layer 7
#     alexnet.add(Dense(4096))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(Dropout(0.5))

#     # Layer 8
#     alexnet.add(Dense(n_classes))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('softmax'))

#     if weights is not None:
#         alexnet.load_weights(weights)

#     return alexnet

# def parse_args():
#     """
#     Parse command line arguments.

#     Parameters:
#         None
#     Returns:
#         parser arguments
#     """
#     parser = argparse.ArgumentParser(description='AlexNet model')
#     optional = parser._action_groups.pop()
#     required = parser.add_argument_group('required arguments')
#     optional.add_argument('--print_model',
#         dest='print_model',
#         help='Print AlexNet model',
#         action='store_true')
#     parser._action_groups.append(optional)
#     return parser.parse_args()

# if __name__ == "__main__":
#     # Command line parameters
#     args = parse_args()

#     # Create AlexNet model
#     model = alexnet_model()
    
#     # from keras.preprocessing.image import load_img
#     # from keras.preprocessing.image import img_to_array

#     # image = load_img('images/dog.jpg', target_size=(224, 224))

#     # # convert the image pixels to a numpy array
#     # image = img_to_array(image)

#     # image = np.expand_dims(image, axis=0)

#     # # prepare the image for the VGG model
#     # input_data = model.predict(image)
#     # # print(input_data)
#     # x = vgg16.VGG16()
#     # label = vgg16.decode_predictions(input_data)
#     # label = label[0][0]
#     # # print the classification
#     # print('%s (%.2f%%)' % (label[1], label[2]*100))
#     # print(model.decode_predictions(input_data))

#     # Print
#     if args.print_model:
#         model.summary()

import keras
from keras.models import Sequential
import keras.applications.vgg16 as vgg16
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)
#Instantiate an empty model

def alexnet(weights=None):

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(1000))
    model.add(Activation('softmax'))
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model

model = alexnet()

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array

# image = load_img('images/dog.jpg', target_size=(224, 224))

# # convert the image pixels to a numpy array
# image = img_to_array(image)

# image = np.expand_dims(image, axis=0)

# # prepare the image for the VGG model
# input_data = model.predict(image)
# # print(input_data)
# x = vgg16.VGG16(weights=None)
# label = vgg16.decode_predictions(input_data)
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))