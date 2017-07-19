from os import listdir, path, getcwd
from PIL import Image
import numpy as np

#constants
height=  50
width = 50

#retrieve all image names and images 
def retrieve_img():
    all_content_names = listdir("contentImages")
    all_style_names = listdir("styleImages")
    
    all_content_images = []
    all_style_images = []
    
    for content in all_content_names:
        image = Image.open('./contentImages/' + str(content)) #retrieve the content image
        all_content_images.append(image)
        
    for style in all_style_names:
        image = Image.open('./styleImages/' + str(style)) #retrieve the style image
        all_style_images.append(image)
        
    return all_content_names,all_style_names,all_content_images,all_style_images

content_names,style_names,content_img,style_img = retrieve_img()

#resize, convert to np array, add placeholder dimension, subtract mean RGB values
def prelim_img_process():
    for each in content_img:
        each = each.resize((height,width), Image.ANTIALIAS) #resize
        each = np.asarray(each, dtype='float32') #cast to np array
        each = np.expand_dims(each,axis=0) #add placeholder dimension
        
        each[:, :, :, 0] -= 103.939 #RGB values obtained from ImageNet
        each[:, :, :, 1] -= 116.779
        each[:, :, :, 2] -= 123.68
        each = each[:, :, :, ::-1]
    

#prelim_img_process()

#print(content_names[0][:-4])

print(getcwd())