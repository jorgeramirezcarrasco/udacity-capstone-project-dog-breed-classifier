import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image, ImageFile
from torchvision import datasets
from torchvision.transforms import ToTensor
from io import BytesIO
import cv2


ImageFile.LOAD_TRUNCATED_IMAGES = True

import boto3
def download_s3_file(BUCKET_NAME, BUCKET_FILE_NAME, LOCAL_FILE_NAME):
    s3 = boto3.client('s3')
    s3.download_file(BUCKET_NAME, BUCKET_FILE_NAME, LOCAL_FILE_NAME)

download_s3_file('sagemaker-eu-central-1-411771656960','capstone-project-dog-breed-classifier/class_names.txt','class_names.txt')
from ast import literal_eval
with open('class_names.txt') as f:
    class_names = literal_eval(f.read())

download_s3_file('sagemaker-eu-central-1-411771656960','capstone-project-dog-breed-classifier/haarcascade_frontalface_alt.xml','haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

VGG16 = models.vgg16(pretrained=True)

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512),
                                  nn.Linear(in_features=512, out_features=256),
                                  nn.Linear(in_features=256, out_features=133))

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'application/json':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def dog_detector(img):
    label = VGG16_predict(img)
    if label > 150 and label < 269:
        return True
    else:
        return False
    

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



def VGG16_predict(img):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img: img
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    transform = transforms.Compose(
    [transforms.Resize(size=(256,256)),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    output = VGG16(transform(img).unsqueeze(0))

    # position 0 return value, position 1 return indices
    return torch.max(output,1)[1].item()

def predict_breed_sagemaker_transfer(img, model):
    # load the image and return the predicted breed
    transform = transforms.Compose(
    [transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    output = torch.from_numpy(model(transform(img).unsqueeze(0)))   
    # position 0 return value, position 1 return indices
    output_index = torch.max(output,1)[1].item()
    return class_names[output_index]

def predict_fn(input_data, model):
    print('Inferring dog breed of input data.')
        
    image_data = re.sub('^data:image/.+;base64,', '', input_data)
    
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    
    if dog_detector(img) is True:
        prediction = predict_breed_sagemaker_transfer(img, model)
        response = "Dogs Detected!\nIt looks like a {0}".format(prediction) 
    elif face_detector(img) > 0:
        prediction = predict_breed_sagemaker_transfer(img, model)
        response = "Hello, human!\nIf you were a dog..You may look like a {0}".format(prediction)
    else:
        response = "Error! Can't detect anything.."
    
    return response
