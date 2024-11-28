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
            image_name = filename[:-4] + ".jpg" # 이미지 파일 이름
            image_path = os.path.join(image_dir, image_name)  # 이미지 파일 경로
            output_path = os.path.join(output_dir, filename[:-4] + ".json") # 출력 JSON 파일 경로


            try:
                # 이미지 크기 가져오기
                with Image.open(image_path) as img:
                    width, height = img.size


                # YOLO 라벨 파일 파싱 및 JSON 데이터 생성
                data = { # json 데이터 딕셔너리
                    "version": "0.4.15",
                    "flags": {},
                    "shapes": [],
                    "imagePath": image_name,
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width
                }

                with open(yolo_path, 'r') as f: # yolo 파일 열기
                    for line in f:
                        parts = line.strip().split()  # 공백으로 분할
                        class_id = parts[0]  # 클래스 ID
                        points = [list(map(float, parts[i:i+2])) for i in range(1, len(parts), 2)] # 폴리곤 좌표
                        # print("points: ", points) # 좌표 확인

                        # AnyLabeling 형식의 shape 데이터 생성
                        shape = {
                            "label": class_id,
                            "text": "",
                            "points": [[x * width, y * height] for x, y in points], # 좌표 스케일링 (정규화된 좌표 -> 픽셀 좌표)
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        }
                        data["shapes"].append(shape)  # shapes 리스트에 추가



                # JSON 파일 저장
                with open(output_path, 'w') as outfile:
                    json.dump(data, outfile, indent=2) # json 파일 저장

                print(f"{filename} 변환 완료: {output_path}")

            except FileNotFoundError: # 파일 못찾으면 예외처리
                print(f"오류: 이미지 파일 {image_path}을 찾을 수 없습니다.")
            except Exception as e: # 기타 예외처리
                print(f"오류: {filename} 변환 실패: {e}")


# 사용 예시:
yolo_dir = "C:/Users/han/Downloads/1_labeled/valid/labels"  # YOLO 라벨 파일 디렉토리
image_dir = "C:/Users/han/Downloads/1_labeled/valid/images" # 이미지 파일 디렉토리
output_dir = "C:/Users/han/Downloads/1_labeled/valid/json"  # 출력 디렉토리



yolo2json_anylabeling(yolo_dir, image_dir, output_dir)