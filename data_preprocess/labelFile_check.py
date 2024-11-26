import os


# 데이터 전처리 검증 코드 
# 1. 각 클래스의 분포정도(불균형 여부 확인)
# 2. 이미지 데이터와 레이블 데이터가 잘 매칭되는지 
# 3. 레이블 데이터에 내용이 누락된건 없는지 
# 4. 레이블 데이터 내 다중객체 존재 여부 확인
# (데이터에 따라서 다중객체가 당연한 경우도 있음)




# 1. 클래스별 이미지 데이터 분포 확인하기(매개변수:최상위 train폴더경로)
def count_class_distribution(train_dir):
    class_counts = {}
    
    # train 디렉토리 내부의 모든 labels 폴더 탐색
    for root, dirs, files in os.walk(train_dir):
        if "labels" in root:  # labels 폴더만 처리
            for label_file in files:
                if label_file.endswith('.txt'):
                    file_path = os.path.join(root, label_file)
                    with open(file_path, 'r') as f:
                        for line in f:
                            class_id = int(line.split()[0])  # 클래스 ID 가져오기
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return class_counts




# 2-1. 이미지 데이터파일과 매칭되는 텍스트 파일이 존재하는지 확인하고
# 2-2. 텍스트 파일내부에 레이블링된 객체의 좌표값이 있는지 확인(빈 파일은 background로 인식되므로 모든 이미지 파일을 잘 가져오기 위함)
def check_missing_and_empty_labels(image_dir, label_dir):   #이미지폴더와 레이블폴더를 모두 확인함
    missing_label_count = 0
    empty_label_count = 0

    # 이미지와 라벨 파일 매칭 확인
    for image_file in os.listdir(image_dir):
        #이미지 파일이 있을 경우
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            #이미지파일명(확장자를 제외한)에 txt를 붙여서 텍스트파일명의 경로 생성
            label_file = os.path.join(label_dir, f"{os.path.splitext(image_file)[0]}.txt")
            #같은 이름의 텍스트 파일이 없다면
            if not os.path.exists(label_file):
                print(f"라벨 파일이 누락된 이미지: {image_file}")
                missing_label_count += 1
            #같은 이름의 텍스트 파일이 있다면
            else:
                with open(label_file, 'r') as f:
                    #텍스트 파일을 읽고
                    lines = f.readlines()
                    #읽을 라인이 없다면
                    if not lines:
                        print(f"빈 라벨 파일: {label_file}")
                        empty_label_count += 1
    return missing_label_count, empty_label_count



# 3. 위 코드 간소화 
# 빈 텍스트파일만 탐지합니다.

def find_empty_label_files(label_dir):
    empty_label_files = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    empty_label_files.append(label_file)
    return empty_label_files



# 4. 텍스트 파일에 여러개의 객체가 존재하는 경우를 찾아냄 
# 의도한 다중 객체를 제외하고 잘못 segmentation된 객체가 있는지 여부를 확인해야 할떄 사용
# ex - 된장찌개만 탐지해야하는데 그릇의 무늬일부를 탐지하여 '된장찌개'라고 레이블링 되어 버리면 학습이 방해가 됨

# 매개변수 min_object를 수정하여 최소 객체의 수를 조절할 수 있음
def find_files_with_multiple_objects(label_dir, min_object = 2):
    """
    텍스트 파일에 다중 객체가 존재하는 경우 탐지.
    :param label_dir: 라벨 폴더 경로
    :param min_objects: 객체 최소 개수 (기본값: 2)
    :return: 다중 객체가 있는 라벨 파일 리스트
    """
    files_with_multiple_objects = []

    # 라벨 디렉토리 내 모든 파일 확인
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'): # 라벨 파일인지 확인
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= min_object:   # 객체가 2개 이상이면
                    files_with_multiple_objects.append(label_file)
    return files_with_multiple_objects





# 실행(필요한 기능 외에 주석처리 후 실행하자)
if __name__ == "__main__":
    
    # 1. 클래스 분포 확인
    train_dir = "C:/Users/Sesame/food_yolo_detection/1st_seg_data_test_241121/valid"
    class_distribution = count_class_distribution(train_dir)
    print("클래스별 샘플 분포:", class_distribution)


    # 2. 누락된/빈 라벨 파일 확인
    image_dir = "C:/Users/Sesame/food_yolo_detection/1st_seg_data_test_241121/train/01011001/images"
    label_dir = "C:/Users/Sesame/food_yolo_detection/1st_seg_data_test_241121/train/01011001/labels"
    missing_count, empty_count = check_missing_and_empty_labels(image_dir, label_dir)
    print(f"누락된 라벨 파일 수: {missing_count}")
    print(f"빈 라벨 파일 수: {empty_count}")


    # 3. 빈 라벨 파일 탐지(2번 함수의 간략한 버전)
    empty_labels = find_empty_label_files(label_dir)
    print(f"\n빈 라벨 파일 목록: {empty_labels}")
    print(f"빈 라벨 파일 수: {len(empty_labels)}")


    # 4. 다중 객체 파일 탐지
    multiple_objects_files = find_files_with_multiple_objects(label_dir)
    print("\n객체가 2개 이상인 라벨 파일:")
    print(multiple_objects_files)
    print(f"객체가 2개 이상인 파일 수: {len(multiple_objects_files)}")