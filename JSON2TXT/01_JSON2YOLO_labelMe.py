import json
import os
import shutil
from pathlib import Path

class SAM_Json2Yolo:
    # 클래스 생성
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cls_names = []


    def convert(self):
        """JSON 파일 로드 > 변환 전처리 실행."""

        ### src > json, img 파일 각각 따로 이동
        self.move_files()

        ## JSON 경로 설정
        json_dir = os.path.join(self.data_dir, 'json')
        json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]

        # JSON 파일 로드
        for json_path in json_files:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)

            # 클래스 이름 추출 > 클래스 이름 저장 (최초 1번만 실행)
            if not self.cls_names:
                self.cls_names = self.extract_cls_names(data)
            self.save_cls_names()

            # 각 이미지 별 라벨 파일 생성
            self.preprocess_img(data)


    def move_files(self):
        """
        원본 디렉토리(data_dir)에 base 폴더 밑에 json, img 폴더 생성 > 각각 폴더로 이동 시키기
        """

        # 폴더 생성
        base_dir = os.path.join(self.data_dir, 'base')
        json_dir = os.path.join(self.base_dir, 'json')
        img_dir = os.path.join(self.base_dir, 'images')
        
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # 파일 이동 (.json > json_dir / .pic > img_dir)
        for filename in os.listdir(self.data_dir):
            src_path = os.path.join(self.data_dir, filename)
            if filename.endswith('.json'):
                shutil.move(src_path, json_dir)
            elif filename.lower().endswith(('.png', 'jpg', 'jpeg', '.bmp')):
                shutil.move(src_path, img_dir)


    def extract_cls_names(self, data):
        """JSON > 클래스 목록으로 추출"""

        clss_names = set()                      # 중복 제거를 위해 set 사용
        for shape in data['shapes']:
            clss_names.add(shape["label"])
        return sorted(list(clss_names))         # 정렬된 리스트로 반환


    def save_cls_names(self):
        """클래스 목록 > .names로 저장"""
        os.makedirs(self.data_dir, exist_ok=True) # 출력 디렉토리 없으면 생성
        with open(os.path.join(self.data_dir, 'classes.names'), 'w') as f:
            for cls_name in self.cls_names:
                f.write(f"{cls_name}\n")


    def preprocess_img(self, data):
        """IMG 경로, 폴리곤 좌표, 클래스ID 매칭 > 라벨링 파일(.txt) 생성"""

        # JSON에서 해당값을 받아옴
        img_name = data['imagePath']
        width = data['imageWidth']
        height = data['imageHeight']

        # img 디렉토리 경로 설정
        img_dir = os.path.join(self.data_dir, "images")

        # 라벨 파일 경로 생성
        label_dir = os.path.join(self.data_dir, 'labels')
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


data_dir = "C:/Github/50_project/test_json4"

# SAM2 > Yolo 변환
converter = SAM_Json2Yolo(data_dir)
converter.convert()