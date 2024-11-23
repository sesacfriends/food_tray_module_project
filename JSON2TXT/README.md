# JSON2TXT

## 구현 계기
   - YOLO v11 Segment 모델에 학습시키기 위해서 라벨링이 필요함
   - Segment는 폴리곤 형식이라 라벨링 난이도가 높고 시간이 오래 걸림
   - 반자동 라벨링 툴을 찾아 라벨링 시작
     - 라벨링 툴: Anylabeling 
     - 라벨링 모델: Meta SAM2
   - 라벨 정보가 JSON 형태로 나옴 > YOLO 학습을 위해서는 txt로 변환 필요

## 1. 레퍼런스 (Reference)
[Ultralytics JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)
기존 코드 (COCO2YOLO)


## 2. 제작 과정 (Version)

### 1st

### 2nd

### 3rd

### 4th
날짜: 2024.11.20 (수)
코드: 2개
01_JSON2YOLO_labelMe.py
  - JSON 파일 정보를 YOLO 파일 정보로 변경
02_verification.py
  - 제대로 변환되었는지 시각화해서 검증

특징: 1개 클래스만 형식 변환 가능 -> 코드 수정 필요

