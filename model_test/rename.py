import os

def rename_files_in_folder(folder_path):

    if not os.path.exists(folder_path):
        print(f"폴더 경로가 존재하지 않습니다: {folder_path}")
        return

    # 폴더 안의 파일 목록 가져오기
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # 정렬 (기본적으로 알파벳 순)

    for i, file_name in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file_name)
        file_extension = os.path.splitext(file_name)[1]  # 확장자 추출
        new_name = f"{i}{file_extension}"  # 새 파일 이름 생성
        new_path = os.path.join(folder_path, new_name)

        # 파일 이름 변경
        os.rename(old_path, new_path)

    print(f"{len(files)}개의 파일 이름이 변경되었습니다.")


folder_path = "real_test_image"  # 파일이 있는 폴더 경로 입력
rename_files_in_folder(folder_path)
