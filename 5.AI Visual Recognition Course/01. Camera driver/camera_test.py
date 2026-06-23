# import cv2
# from picamera2 import Picamera2

# try:
#   # For webcam input:
#   picam2 = Picamera2()
#   picam2.configure(
#       picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
#   )
#   picam2.start()

#   while True:
#     frame = picam2.capture_array() 
#     frame = cv2.flip(frame, 1)
    
#     cv2.imshow("image1",frame)

#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# except:
#   picam2.stop()
#   picam2.close()

import cv2
import os
from ultralytics import YOLO

# 현재 실행 중인 스크립트 파일(.py)의 절대 경로 폴더 알아내기
current_dir = os.path.dirname(os.path.abspath(__file__))
# 현재 폴더 경로에 "best.pt" 파일명 결합
model_path = os.path.join(current_dir, "recycle_person.pt")

# 동적으로 생성한 절대 경로를 사용해 모델 로드
model = YOLO(model_path)

# 노트북 기본 웹캠 연결
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # AI 모델이 분석하기 좋게 해상도 확장
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 불러올 수 없습니다.")
            break
        
        # 거울 모드 (좌우 반전)
        frame = cv2.flip(frame, 1)
        
        # 2. YOLO11 모델에게 프레임을 던져주고 인식 결과(박스, 확률)를 받아오기
        # conf=0.5 옵션: AI가 50% 이상 확신하는 객체만 화면에 표시
        results = model.predict(source=frame, show=False, conf=0.75)
        
        # 3. 인식 결과(테두리 박스와 이름)가 그려진 이미지 데이터 추출
        annotated_frame = results[0].plot()
        
        # 4. 화면 출력
        cv2.imshow("YOLO11 Object Detection", annotated_frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
