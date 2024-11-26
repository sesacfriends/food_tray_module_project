import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np


# 1. 모델 경로 및 validation 데이터 경로 설정
model_path = 'best_pt/best.pt'  # 학습된 YOLO 모델 파일 경로
valid_folder = 'C:/Users/Sesame/food_yolo_detection/1st_seg_data_test_241121/valid'  # Validation 데이터 상위 폴더 경로
classes = sorted(os.listdir(valid_folder))  # Validation 폴더 안의 클래스 폴더 이름들을 정렬하여 리스트로 반환

# 2. 클래스별 색상 정의 (6개 클래스)
colors = [
    (255, 0, 0),   # Class 0 - 빨간색
    (0, 255, 0),   # Class 1 - 초록색
    (0, 0, 255),   # Class 2 - 파란색
    (255, 255, 0), # Class 3 - 하늘색
    (255, 0, 255), # Class 4 - 자홍색
    (0, 255, 255)  # Class 5 - 노란색
]

# 3. YOLO 모델 로드
model = YOLO(model_path)  # 학습된 모델 불러오기

# 4. 잘못 분류된 이미지 정보를 저장할 데이터 구조
misclassified_images = []  # 잘못 분류된 이미지 데이터를 저장 (이미지, 실제 클래스, 예측 클래스)
misclassified_counts = {}  # 클래스별 잘못 분류된 이미지 개수를 저장 (딕셔너리 형태)

# 5. Validation 데이터 처리
for label_idx, class_name in enumerate(classes):  # 각 클래스 폴더를 순회하며 idx와 클래스 이름 가져오기
    print(f'classes : {classes}')   
    print(f'label_idx : {label_idx}')
    print(f'class_name : {class_name}')

    # 각 클래스 폴더 안의 'images' 서브폴더 경로 설정
    class_image_folder = os.path.join(valid_folder, class_name, 'images')
    if not os.path.exists(class_image_folder):  # 해당 폴더가 없으면 다음 클래스로 넘어감
        continue

    # 해당 클래스의 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(class_image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 각 이미지 파일에 대해 예측 수행
    for image_file in image_files:
        # 이미지 파일 경로 설정 및 읽기
        image_path = os.path.join(class_image_folder, image_file)
        image = cv2.imread(image_path)  # 이미지를 BGR 형식으로 읽음 (OpenCV 기본 설정)

        # YOLO 모델을 사용하여 예측 수행
        results = model(image)

        # 6. Segmentation 마스크 처리
        if results[0].masks is not None:  # Segmentation 마스크 결과가 있는 경우만 처리
            masks = results[0].masks.data.cpu().numpy()  # (N, H, W) 형태의 binary masks로 변환
            boxes = results[0].boxes  # 탐지된 바운딩 박스 정보 (클래스 ID 포함)
            cls_ids = []  # 탐지된 객체의 클래스 ID를 저장할 리스트

            for box in boxes:  # 탐지된 각 바운딩 박스를 순회
                cls_id = int(box.cls.item())  # 각 박스의 클래스 ID 추출
                cls_ids.append(cls_id)  # 추출한 클래스 ID를 리스트에 추가

            # 각 마스크를 이미지에 overlay
            if label_idx not in cls_ids:  # 현재 클래스가 탐지 결과에 포함되지 않은 경우
                for mask, cls_id in zip(masks, cls_ids):
                    mask = (mask * 255).astype(np.uint8)  # 0~1 스케일을 0~255로 변환
                    if int(cls_id) < len(colors):
                        color = colors[int(cls_id)]  # 클래스별 색상 선택
                    else: 
                        color = (255, 255, 255)  # 색상 정보가 없을 경우 흰색 사용

                    # 색상 마스크 생성
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[mask > 0] = color  # 마스크가 있는 영역에만 색상 적용

                    # 원본 이미지와 합성
                    image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        
        # 모델의 예측값 추출
        pred_classes = []  # 탐지된 클래스 ID를 저장할 리스트
        for box in results[0].boxes:
            pred_class = int(box.cls.item())  # 각 바운딩 박스의 클래스 ID
            pred_classes.append(pred_class)

        # 7. 잘못 분류된 경우 처리
        if label_idx not in pred_classes:  # 실제 클래스가 탐지된 예측값에 포함되지 않은 경우
            misclassified_images.append((image, class_name, pred_classes))  # 잘못된 이미지 정보 저장
            if class_name in misclassified_counts:
                misclassified_counts[class_name] += 1  # 클래스별 카운트 증가
            else:
                misclassified_counts[class_name] = 1  # 새로운 클래스라면 초기화

# 8. 잘못 분류된 이미지 시각화 함수
def plot_misclassified(images_with_labels):
    """잘못 분류된 이미지를 n행 × n열 그리드로 시각화"""
    total_images = len(images_with_labels)  # 잘못 분류된 전체 이미지 개수
    cols = 4  # 한 줄에 표시할 이미지 수
    rows = (total_images // cols) + 1 if total_images % cols != 0 else total_images // cols  # 행 계산

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Matplotlib 서브플롯 생성
    for i, (img, true_label, pred_labels) in enumerate(images_with_labels):
        ax = axes.flatten()[i] if i < rows * cols else None
        if ax:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 이미지를 BGR에서 RGB로 변환하여 표시
            pred_text = ', '.join(classes[p] for p in pred_labels)  # 예측 클래스 이름 조합
            ax.set_title(f"True: {true_label}\nPred: {pred_text}")  # 실제 클래스와 예측 클래스 표시
            ax.axis('off')  # 축 숨기기
    # 빈 플롯 비활성화
    for j in range(i + 1, rows * cols):
        axes.flatten()[j].axis('off')
    plt.tight_layout()  # 레이아웃 조정
    plt.show()

# 9. 테스트 실행
if len(misclassified_images) > 0:  # 잘못 분류된 이미지가 있는 경우
    plot_misclassified(misclassified_images)  # 시각화 함수 호출
else:
    print("모든 Validation 데이터가 올바르게 분류되었습니다.")  # 모든 데이터가 올바르게 분류된 경우 메시지 출력

# 10. 잘못 분류된 전체 이미지 개수 출력
print(f"잘못 분류된 이미지 총 개수: {len(misclassified_images)}")

# 11. 클래스별 잘못 분류된 이미지 개수 출력
print('잘못 분류된 이미지 개수 (클래스별):')
for class_name, count in misclassified_counts.items():  # 각 클래스별 개수 출력
    print(f'{class_name} : {count}')
print(f'misclassified_counts : {misclassified_counts}')  # 전체 딕셔너리 출력