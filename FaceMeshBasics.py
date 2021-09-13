import cv2
import mediapipe as np
import time

cap = cv2.VideoCapture("Videos/2.mp4")
pTime = 0

mpDraw = np.solutions.drawing_utils
mpFaceMesh = np.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# MAX_NUM_FACES : 감지할 최대 얼굴 수입니다. 기본값은 1입니다.
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
# fhd일 경우 값 변경해서 더 잘보이게 변경할 수 있습니다

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.cvtColor,
                                  drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                '''
                print(lm) #landmark 번호 
                x: 0.4825424253940582
                y: 0.30031031370162964
                z: -0.00373118557035923
                '''
                ih, iw, ic = img.shape #image height, width, channel
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
                '''
                0~467까지 총 468 lm (x,y)를 찍습니다
                467 1066 326
                0 979 462
                1 978 421
                2 978 432
                ...
                '''

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 3)
    cv2.imshow("Images", img)
    cv2.waitKey(1)