import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# 1. 모델 경로 정의
model_path = 'best_pt/best.pt'  # 학습된 YOLO 모델 파일 경로

# 2. 테스트할 이미지 파일 경로 정의
image_folder = 'real_test_image'  # 테스트할 이미지들이 저장된 폴더 경로
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# image_files: 주어진 폴더에서 확장자가 '.jpg', '.jpeg', '.png'인 파일만 가져와 리스트로 저장

# 클래스별 색상 정의 (6개 클래스)
# Segmentation 결과 시각화를 위해 각 클래스에 고유 색상 할당
colors = [
    (255, 0, 0),   # Class 0 - Red
    (0, 255, 0),   # Class 1 - Green
    (0, 0, 255),   # Class 2 - Blue
    (255, 255, 0), # Class 3 - Cyan
    (255, 0, 255), # Class 4 - Magenta
    (0, 255, 255)  # Class 5 - Yellow
]

# 모델 불러오기
if os.path.exists(model_path):  # 모델 파일이 존재하는지 확인
    model = YOLO(model_path)  # YOLO 모델 로드
    model_name = os.path.basename(os.path.dirname(model_path))  # 모델 경로에서 폴더 이름 추출 (예: 'best_pt')

    # 예측 결과를 저장할 리스트
    processed_images = []  # (이미지, 제목) 형태의 튜플 리스트로 저장

    # 각 이미지에 대해 예측 수행
    for image_file in image_files:  # 테스트할 각 이미지 파일에 대해 반복
        image_path = os.path.join(image_folder, image_file)  # 이미지 파일의 전체 경로
        image = cv2.imread(image_path)  # 이미지 읽기 (OpenCV로 BGR 형식으로 읽힘)
        resized_image = cv2.resize(image, (640, 640))  # 이미지를 640x640으로 리사이즈

        # 모델 예측 수행
        results = model(resized_image)  # YOLO 모델로 이미지 예측 수행

        # 예측 결과 시각화
        for result in results:  # 결과 리스트 순회
            if result.masks is None:  # Segmentation 마스크가 없는 경우
                print("탐지되는 객체 없음")  # 디버깅 출력
                continue  # 다음 이미지로 넘어감

            masks = result.masks.data.cpu().numpy()  # 마스크 데이터를 numpy 배열로 변환 (N, H, W 형태)
            boxes = result.boxes  # 탐지된 객체의 바운딩 박스 정보 (클래스 ID, confidence 포함)

            # Segmentation 결과를 RGB 마스크로 변환
            segmentation_image = np.zeros_like(resized_image, dtype=np.uint8)  # 원본 이미지와 동일한 크기의 빈 RGB 이미지 생성
            for i, mask in enumerate(masks):  # 각 객체에 대해 마스크 처리
                class_id = int(boxes.cls[i].item())  # 탐지된 객체의 클래스 ID 추출
                confidence = boxes.conf[i].item()  # 탐지된 객체의 confidence score 추출
                color = colors[class_id]  # 클래스 ID에 따라 미리 정의된 색상 선택

                # 마스크를 해당 클래스 색상으로 채우기
                segmentation_image[mask > 0.5] = color  # 마스크가 0.5 이상인 픽셀만 색상 적용

            # 원본 이미지와 마스크 합성 (오버레이)
            overlay_image = cv2.addWeighted(resized_image, 0.7, segmentation_image, 0.3, 0)
            # cv2.addWeighted: 두 이미지를 가중치로 합성 (원본 이미지 70%, 마스크 이미지 30%)

            # 각 객체에 대해 클래스 이름과 confidence score 텍스트 추가
            for i, mask in enumerate(masks):
                class_id = int(boxes.cls[i].item())  # 클래스 ID
                confidence = boxes.conf[i].item()  # confidence score

                # 마스크 영역의 중심 좌표 계산
                y_coords, x_coords = np.where(mask > 0.5)  # 마스크의 픽셀 좌표 (y, x) 추출
                if len(x_coords) > 0 and len(y_coords) > 0:  # 마스크 영역이 존재하는 경우
                    x_center = int(np.mean(x_coords))  # x 좌표 중심 계산
                    y_center = int(np.mean(y_coords))  # y 좌표 중심 계산
                    text = f"{model.names[class_id]}: {confidence:.2f}"  # 클래스 이름과 confidence score 텍스트 생성
                    # 텍스트를 이미지에 추가 (OpenCV 사용)
                    cv2.putText(overlay_image, text, (x_center, y_center),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            # 결과 이미지를 RGB로 변환하여 저장
            processed_images.append((cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB), f"{model_name} - {image_file}"))


# 3. 여러 이미지를 n행 × n열 그리드로 시각화하는 함수 정의
def plot_grid(images_with_titles, rows, cols, grid_title):
    """이미지를 숫자 기준으로 정렬한 후 n행 × n열 그리드로 표시, 그리드 제목 포함"""
    # 숫자를 기준으로 정렬
    images_with_titles_sorted = sorted(
        images_with_titles,
        key=lambda x: int(''.join(filter(str.isdigit, x[1]))),  # 제목에서 숫자 추출 후 정렬
    )

    # Matplotlib 서브플롯 생성
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # n행 × n열 크기의 서브플롯 생성

    # 각 이미지에 대해 축 설정
    for i, (img, title) in enumerate(images_with_titles_sorted):  # 정렬된 이미지 순회
        ax = axes.flatten()[i] if i < rows * cols else None  # 플롯 인덱스 설정
        if ax:
            ax.imshow(img)  # 이미지 표시
            ax.set_title(title.split(' ')[-1])  # 파일명 표시
            ax.axis('off')  # 축 숨기기

    # 그리드 전체의 제목 설정
    fig.suptitle(grid_title, fontsize=16)  # 제목 설정

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 제목 공간 확보
    plt.show()


# 4. 예측 결과를 그리드로 시각화
total_images = len(processed_images)  # 처리된 이미지 개수
cols = 4  # 한 줄에 표시할 이미지 수
rows = (total_images // cols) + 1 if total_images % cols != 0 else total_images // cols  # 행 수 계산

# 예측 결과를 그리드로 시각화, 그리드에 모델 이름을 제목으로 설정
plot_grid(processed_images, rows=rows, cols=cols, grid_title=f"Model: {model_name}")
