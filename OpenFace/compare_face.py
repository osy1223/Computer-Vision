import face_recognition
from face_recognition.api import face_encodings
import matplotlib.pyplot as plt

img1 = face_recognition.load_image_file('5-celebrity-faces-dataset/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')
plt.imshow(img1)
plt.show()

img2 = face_recognition.load_image_file('5-celebrity-faces-dataset/data/train/elton_john/httpftqncomymusicLxZeltonjohnjpg.jpg')
plt.imshow(img2)
plt.show()

img1_face_encodings = face_recognition.face_encodings(img1)[0]
img2_face_encodings = face_recognition.face_encodings(img2)[0]

result = face_recognition.compare_faces([img1_face_encodings],img2_face_encodings)
if result[0]:
    print('동일인물 입니다')
else:
    print('동일인물 아닙니다')
