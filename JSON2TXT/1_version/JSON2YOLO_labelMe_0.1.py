import json
import os
from pathlib import Path

class SAM2Json2Yolo:
    # 클래스 생성
    def __init__(self, json_path, img_dir, output_dir):
        self.json_path = json_path
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.cls_names = []


    def convert(self):
        """JSON 파일 로드 > 변환 전처리 실행."""
        # JSON 파일 로드
        with open(self.json_path) as f:
            data = json.load(f)

        # 클래스 이름 추출 > 클래스 이름 저장
        self.cls_names = self.extract_cls_names(data)
        self.save_cls_names()

        # 각 이미지 별 라벨 파일 생성
        self.preprocess_img(data)


    def extract_cls_names(self, data):
        """JSON > 클래스 목록으로 추출"""

        clss_names = set()                      # 중복 제거를 위해 set 사용
        for shape in data['shapes']:
            clss_names.add(shape["label"])
        return sorted(list(clss_names))         # 정렬된 리스트로 반환


    def save_cls_names(self):
        """클래스 목록 > .names로 저장"""
        os.makedirs(self.output_dir, exist_ok=True) # 출력 디렉토리 없으면 생성
        with open(os.path.join(self.output_dir, 'classes.names'), 'w') as f:
            for cls_name in self.cls_names:
                f.write(f"{cls_name}\n")


    def preprocess_img(self, data):
        """IMG 경로, 폴리곤 좌표, 클래스ID 매칭 > 라벨링 파일(.txt) 생성"""
        img_name = data['imagePath']
        width = data['imageWidth']
        height = data['imageHeight']

        # 라벨 파일 경로 생성
        label_dir = os.path.join(self.output_dir, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        label_file_path = os.path.join(label_dir, f"{Path(img_name).stem}.txt")

        with open(label_file_path, 'w') as label_file:
            for shape in data['shapes']:

                # 클래스 아이디
                cls_id = self.cls_names.index(shape['label'])

                # 폴리곤 좌표 > yolo형식으로 변환
                    # SAM2 좌표형식: [x1, y1], [x2, y2]...
                    # YOLO 좌표형식: [z1 z2 z3 z4...]
                polygon_points = shape['points']

                    # SAM2 좌표 > 문자열로 변환
                polygon_str = ""
                for x, y in polygon_points:
                     polygon_str += f"{x / width:.6f} {y / height:.6f} "

                # 클래스 아이디, 폴리곤 좌표 > 파일에 write
                    # split()으로 공백 제거 > join으로 공백 1칸 통일
                label_file.write(f"{cls_id} {' '.join(polygon_str.split())}\n")


JsonPath = "C:/Github/50_project/test_json/src"
imgPath = "C:/Github/50_project/test_json/src"
OutPath = "C:/Github/50_project/test_json/out"

# SAM2 > Yolo 변환
converter = SAM2Json2Yolo(JsonPath, imgPath, OutPath)
converter.convert()