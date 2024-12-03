import json
import os
from pathlib import Path
from PIL import Image

def yolo2json_anylabeling(yolo_dir, image_dir, output_dir):
    """
    YOLO 형식의 segmentation 라벨 파일(.txt)을 AnyLabeling JSON 형식으로 변환

    Args:
        yolo_dir (str): YOLO 라벨 파일이 있는 디렉토리 경로
        image_dir (str): 이미지 파일이 있는 디렉토리 경로
        output_dir (str): 출력 디렉토리 경로
    """

    os.makedirs(output_dir, exist_ok=True) # 출력 디렉토리 생성

    for filename in os.listdir(yolo_dir): # yolo_dir 파일 목록 순회
        if filename.endswith(".txt"): # .txt 파일만 처리
            yolo_path = os.path.join(yolo_dir, filename) # yolo 파일 경로
            base_name = filename[:-4]  # 파일 이름 (확장자 제외)
            # image_name = filename[:-4] + ".jpg" # 이미지 파일 이름
            # image_path = os.path.join(image_dir, image_name)  # 이미지 파일 경로
            # output_path = os.path.join(output_dir, filename[:-4] + ".json") # 출력 JSON 파일 경로 # 확장자 제외

            # 다양한 이미지 확장자 처리
            image_path = None
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
                temp_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    image_name = base_name + ext # 이미지 파일 이름 (확장자 포함)
                    break

            if image_path is None:  # 이미지 파일을 찾지 못한 경우
                            print(f"오류: {base_name}에 해당하는 이미지 파일을 찾을 수 없습니다.")
                            continue

            output_path = os.path.join(output_dir, base_name + ".json")

            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                data = {
                    "version": "0.4.15",
                    "flags": {},
                    "shapes": [],
                    "imagePath": image_name,  # 확장자 포함 이미지 파일 이름 사용
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width
                }

                with open(yolo_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = parts[0]
                        points = [list(map(float, parts[i:i+2])) for i in range(1, len(parts), 2)]

                        shape = {
                            "label": class_id,
                            "text": "",
                            "points": [[x * width, y * height] for x, y in points],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        }
                        data["shapes"].append(shape)

                with open(output_path, 'w') as outfile:
                    json.dump(data, outfile, indent=2)

                print(f"{filename} 변환 완료: {output_path}")

            except Exception as e:
                print(f"오류: {filename} 변환 실패: {e}")



# 사용 예시:
yolo_dir = "C:/Users/han/Downloads/1st_seg_data_test_241121/1st_seg_data_test_241121/train/07014001/labels"  # YOLO 라벨 파일 디렉토리
image_dir = "C:/Users/han/Downloads/1st_seg_data_test_241121/1st_seg_data_test_241121/train/07014001/temp_ing" # 이미지 파일 디렉토리
output_dir = "C:/Users/han/Downloads/1st_seg_data_test_241121/1st_seg_data_test_241121/train/07014001/jason"  # 출력 디렉토리



yolo2json_anylabeling(yolo_dir, image_dir, output_dir)