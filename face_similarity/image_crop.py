



from PIL import Image

path = './test/img5.jpg'



img = Image.open(path)
# img.show()

# print(img.size) # (512, 512)


# 이미지의 일부분을 얻고 싶을 때는 crop( ) 함수를 사용합니다.
# 마찬가지로 인자를 튜플 형태 (left, upper, right, lower)로 입력 받습니다.
# 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
# croppedImage=image1.crop((10,10,100,100))

# 상자의 모든 좌표 (x, y, w, h)는 이미지의 왼쪽 상단에서 측정됩니다.
# 따라서 상자의 좌표는 (x, y, w + x, h + y) 여야합니다. 코드를 다음과 같이 변경하십시오.

# ((190, 208), (316, 207))
eye_crop = img.crop((100,150,400,250))
# eye_crop.show()
# eye_save = eye_crop.save('./eyes/eye_crop_1.jpg')
# print('saved')




nose_crop = img.crop((220,200,290,300))
# nose_crop.show()
# nose_save = nose_crop.save('./nose/nose_crop_4.jpg')
# print('saved')


mouth_crop = img.crop((200,300,300,400))
mouth_crop.show()
mouth_save = mouth_crop.save('./mouth/mouth_crop_4.jpg')
print('saved')




# print(img.size)

# from mtcnn.mtcnn import MTCNN

# detector = MTCNN()


# import numpy as np

# img.convert('RGB')
# img = np.asarray(img)
# test = detector.detect_faces(img)
# print(test)


# landmarks = test[0]['keypoints']
# print(landmarks)

# eyes = landmarks['left_eye'],landmarks['right_eye']
# print(eyes)



