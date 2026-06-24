import os
import glob
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

# 윈도우 환경에서 YOLO 학습 시 멀티프로세싱 충돌을 막기 위한 필수 보호 구문입니다.
if __name__ == '__main__':

    # 1. 데이터셋 다운로드 (내 컴퓨터로 다운로드 됨)
    rf = Roboflow(api_key="Qyyw9pUS3kKjLZMAxmY5")
 
    print("📦 분리수거 데이터셋 다운로드 중...")
    recycle_project = rf.workspace("recycle-sx0ka").project("recycle-trnfc")
    recycle_dataset = recycle_project.version(22).download("yolov11")

    print("📦 사람(배경) 데이터셋 다운로드 중...")
    person_project = rf.workspace("project-1l5rc").project("person-g4qsf")
    person_dataset = person_project.version(4).download("yolov11")

    # 2. 파일 제어를 통해 사람 사진만 분리수거 이미지 폴더로 이동하는 함수
    def merge_person_as_background(src_dir, dst_dir, max_count=100):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        person_images = []
        for ext in exts:
            person_images.extend(glob.glob(os.path.join(src_dir, ext)))
        
        count = 0
        for img_path in person_images[:max_count]:
            # 파일이 이미 존재하면 덮어쓰거나 무시하도록 에러 처리 가능하지만, 일반 복사 사용
            shutil.copy(img_path, dst_dir)
            count += 1
        print(f"✅ {dst_dir} 경로로 {count}장의 사람 이미지가 병합되었습니다.")

    # 학습(train) 및 검증(valid) 폴더에 사람 사진 병합
    merge_person_as_background(
        src_dir=os.path.join(person_dataset.location, "train", "images"),
        dst_dir=os.path.join(recycle_dataset.location, "train", "images"),
        max_count=100
    )
    merge_person_as_background(
        src_dir=os.path.join(person_dataset.location, "valid", "images"),
        dst_dir=os.path.join(recycle_dataset.location, "valid", "images"),
        max_count=20
    )

    # 3. 모델 로드 및 GTX 1650 맞춤형 학습 시작
    print("🚀 내 컴퓨터의 GPU를 사용하여 AI 학습을 시작합니다...")
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=f"{recycle_dataset.location}/data.yaml",
        epochs=70, 
        patience=20,     
        
        # ⚠️ [중요] GTX 1650 (4GB VRAM) 최적화 세팅
        imgsz=640,       # 해상도 하향 조정 (1024 -> 640)
        batch=4,         # 배치 사이즈 하향 조정 (16 -> 4)
        device=0,        # 내장된 NVIDIA GPU 사용
        workers=0,       # 윈도우 환경 멀티프로세싱 충돌 방지
        
        # === 데이터 증강(Augmentation) 파라미터 ===
        hsv_v=0.2,       
        degrees=10.0,    
        translate=0.1,   
        fliplr=0.5,      
        mosaic=0.0,      
        erasing=0.0      
    )
    print("✅ 학습이 완료되었습니다! 현재 폴더의 runs/detect/train/weights 에서 best.pt를 확인하세요.")