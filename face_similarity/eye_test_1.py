
path1 = './mouth/mouth_crop_1.jpg'
path2 = './mouth/mouth_crop_2.jpg'

from cmath import cos, pi
from PIL import Image
import numpy as np


img1 = Image.open(path1)
# print(img1)
img_resize1 = img1.resize((160,160))
# print(img1)
# print(img1)
# img1 = img1.convert('RGB')
# print(img1)
# pixels1 = np.asarray(img1)
# print(pixels1)


img2 = Image.open(path2)
img_resize2 = img2.resize((160,160))




imgs1 = np.array(img_resize1)
imgs2 = np.array(img_resize2)

# print('11111111111111111111',imgs1.shape) # (47, 56, 4)
# img = imgs.reshape(-1,112,112,3) 
img1 = imgs1.reshape(160,160,3) 
img2 = imgs2.reshape(160,160,3)
# print(img1)


#Generalize the data and extract the embeddings
def extract_embeddings(model,face_pixels):
    #   face_pixels = face_pixels.astype('float32')  #convert the entire data to float32(base)
    data = np.array(face_pixels, dtype=np.float32)
    # print('data',data)

    mean = face_pixels.mean()                    #evaluate the mean of the data
    std  = face_pixels.std()                     #evaluate the standard deviation of the data
    face_pixels = (face_pixels - mean)/std       
    samples = np.expand_dims(face_pixels,axis=0)    #expand the dimension of data 
    yhat = model.predict(samples)
    return yhat[0]  


# data = np.array(data, dtype=np.float32)

import arcface_model
import facenet_model

# model = arcface_model.loadModel()
model = facenet_model.loadModel()

embedding1 = extract_embeddings(model, img1)
# print(embedding1)
embedding2 = extract_embeddings(model, img2)

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

# cosine_dis = cosine_distances([embedding1], [embedding2])
# print(cosine_dis) # [[0.08721882]]

cosine_sim = cosine_similarity([embedding1], [embedding2])
print(cosine_sim) 
# [[0.9127812]] arcface
# [[0.6051124]] facenet