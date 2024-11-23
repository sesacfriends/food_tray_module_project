import os
import re

def find_files_with_large_numbers(directory):
    """
    YOLO형식에서 벗어난 파일 찾기
    디렉토리 내의 모든 텍스트 파일을 검사
    1.0보다 큰 숫자가 있는 파일을 출력

    Args:
        directory (str): 검사할 디렉토리 경로
    """
    for filename in os.listdir(directory): # 디렉토리 내 모든 파일 순회
        if filename.endswith(".txt"): # .txt 파일만 처리
            file_path = os.path.join(directory, filename) # 파일 경로
            try:
                with open(file_path, 'r') as f: # 파일 열기
                    for line in f: # 각 라인에 대해 반복
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)  # 숫자 추출 (정수, 부동 소수점 모두)
                        for number in numbers: # 추출된 숫자들에 대해 반복
                            if float(number) > 1.0: # 숫자가 1.0보다 큰 경우
                                print(f"파일 {filename}에 1.0보다 큰 숫자 {number}이 있습니다.")
                                break  # 파일당 한 번만 출력
            except Exception as e:
                print(f"파일 {filename}을 읽는 중 오류 발생: {e}")


# 사용 예시:
directory = "C:/Github/50_project/labels"  # 텍스트 파일이 있는 디렉토리 경로로 변경
find_files_with_large_numbers(directory)