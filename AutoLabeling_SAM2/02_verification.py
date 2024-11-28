import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_segmentation(image_path, label_path, class_names):
    """
    JSON2YOLO 검증
    이미지와 segmentation 마스크 시각화
    """

    # 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    height, width, _ = img.shape

    # 마스크 이미지 생성
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 마스크 색상 선택
        # 클래스명 지정
    mask_colors = { # class_names : colors 로 딕셔너리 생성
        'bob': (0, 0, 255),    # 빨간색
        'guk': (255, 0, 0),   # 파란색
        'snack': (0, 255, 255)   # 노란색
        }
    
    # 라인 색상/선 두께 지정
    line_colors = (255, 255, 255)   # 하얀색
    thickness = 2

    # 라벨 파일에서 폴리곤 좌표와 클래스 ID 읽어오기
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            class_name = class_names[class_id]  # 클래스 이름 가져오기

            polygon_points = np.array(list(map(float, parts[1:]))).reshape(-1, 2)  # 폴리곤 좌표
            polygon_points[:, 0] *= width  # 너비 스케일링
            polygon_points[:, 1] *= height  # 높이 스케일링
            polygon_points = polygon_points.astype(np.int32)  # 정수형으로 변환

            # 폴리곤 그리기 (마스크 생성)
                # 위에서 지정한 클래스별 색상 가져오기 (없으면 랜덤 색상)
            color = mask_colors.get(class_name, np.random.randint(0, 255, size=3).tolist())  # 랜덤 색상
            cv2.fillPoly(mask, [polygon_points], color)
                # 외각선 추가
            cv2.polylines(img, [polygon_points], isClosed=True, color=line_colors, thickness=thickness)

            x, y = polygon_points[0] # 첫번째 좌표
            cv2.putText(img, class_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    # 원본 이미지와 마스크 이미지 합치기 (투명도 조절)
    alpha = 0.5  # 투명도 (0 ~ 1)
    result_image = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)


    # 시각화
    plt.imshow(result_image)
    plt.title(Path(image_path).name)
    plt.show()


# 사용 예시
image_file = "C:/Users/han/Desktop/validation/2nd_dataset(resized)/images-20241127T091223Z-001/images/04_041_04011007_160273723933160.jpg"  # 이미지 파일 경로
label_file = "C:/Users/han/Desktop/validation/2nd_dataset(resized)/labels-20241127T095729Z-001_2nd_valid/04_041_04011007_160273723933160.txt"  # 라벨 파일 경로
class_names = ['04011007'] # 클래스 이름 목록
visualize_segmentation(image_file, label_file, class_names)