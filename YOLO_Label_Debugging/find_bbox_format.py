import os

def find_files_with_few_numbers(directory):
    """
    BBOX형식의 파일 이름 출력
    디렉토리 내의 모든 텍스트 파일을 검사
    BBOX형식 기준: 공백으로 구분된 숫자가 6개 미만
    (예) 0 0.345 0.234 0.675 0.849

    Args:
        directory (str): 검사할 디렉토리 경로
    """
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # .txt 파일만 처리
            file_path = os.path.join(directory, filename) # 파일 경로
            try:
                with open(file_path, 'r') as f: # 파일 열기
                    for line in f:  # 각 라인에 대해 반복
                        numbers = line.strip().split()  # 공백을 기준으로 분할하여 숫자 리스트 생성
                        if len(numbers) < 6: # 숫자 개수가 6개 미만인 경우
                            print(f"파일 {filename}의 라인 '{line.strip()}'에는 숫자가 6개 미만입니다.")
                            break # 파일당 한 번만 출력

            except Exception as e: # 예외 처리
                print(f"파일 {filename}을 읽는 중 오류 발생: {e}")



# 사용:
directory = "C:/Github/50_project/labels"  # 텍스트 파일이 있는 디렉토리 경로로 변경
find_files_with_few_numbers(directory)