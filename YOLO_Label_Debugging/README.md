# AutoLabeling_SAM2

## 구현 계기
   - YOLO v11 Segment 라벨링을 AnyLabeling 툴에 META SAM2를 활용해서 진행중임
   - 기존 BBOX 라벨링과 비슷한 속도로 라벨링 가능
   - 하지만 각 클래스 별로 1,400장의 라벨링이 필요하여 각 클래스당 라벨링 작업 소요시간 3~4시간임
   - 이에 라벨링 시간을 줄이기 위해 SAM2 모델 활용방법을 제안함
   - SAM2 모델 활용방법
      A. 모델을 학습 시켜서 라벨링하기 (fine tuning/ few shot)
      B. 모델에 사진을 넣어서 라벨링하기 (zero shot)
        - 이 코드는 B방식임

## 1. 레퍼런스 (Reference)
[rogoflow_blog](https://blog.roboflow.com/label-data-with-grounded-sam-2/)
[roboflow_blog_KR](https://velog.io/@hsbc/How-to-Label-Data-with-Grounded-SAM-2)
[Ultralytics_Document](https://docs.ultralytics.com/ko/models/sam-2/)

## 2. 과정 (Version)
[팀 Notion 참조](https://www.notion.so/wk-user-manual/SAM2-2-c5ab1cc290aa4081ad80565ed520be26?pvs=4)

