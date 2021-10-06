from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd

img1_path = 'tests/dataset/img1.jpg'
img2_path = 'tests/dataset/img15.jpg'

img1 = DeepFace.detectFace(img1_path)
img2 = DeepFace.detectFace(img2_path)

plt.imshow(img1)
#plt.show()

plt.imshow(img2)
#plt.show()

model_name = 'Ensemble'

resp = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name=model_name)

#print(resp)
#{'verified': False, 'score': 0.9984606326447921, 'distance': [0.6558067208900678, 0.9212336832115025, 1.145256932648799, 1.118929959933359, 17.59915862256838, 1.495947833270505, 0.6737322575183666, 1.1608033920680683, 0.33457734343832457, 74.32064418163229, 0.8180187570445125], 'model': ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace'], 'similarity_metric': ['cosine', 'euclidean', 'euclidean_l2']}


# 안젤리나 졸리 이미지
#df = DeepFace.find(img_path='C:/Users/wing7/Desktop/source.jpg', db_path = 'deepface/tests/dataset')
# VGG_face : find function lasts  0.12094449996948242  seconds
# OpenFace : find function lasts  5.387073278427124  seconds
# Facenet : find function lasts  5.371787786483765  seconds
# Fancenet512 : find function lasts  5.818359613418579  seconds
# DeepFace : find function lasts  5.444867134094238  seconds
# DeepID : find function lasts  5.8870158195495605  seconds
# ArcFace : find function lasts  5.5590009689331055  seconds

#print(df.head())
'''
                            identity  VGG-Face_cosine
0   deepface/tests/dataset//img4.jpg         0.209358
1   deepface/tests/dataset//img5.jpg         0.213314
2   deepface/tests/dataset//img2.jpg         0.263704
3   deepface/tests/dataset//img6.jpg         0.286422
4  deepface/tests/dataset//img11.jpg         0.317846
'''

#model_name = 'Dlib'

#obj = DeepFace.analyze(img_path=img1_path)
#print(obj)
