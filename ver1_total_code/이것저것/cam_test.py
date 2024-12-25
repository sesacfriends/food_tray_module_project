import cv2

def test_camera():
    # 카메라 초기화 (0번 카메라 사용) 
    cap = cv2.VideoCapture(2)

    # 카메라가 제대로 열렸는지 확인
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Camera is working. Press ESC to exit.")

    while True:
        # 카메라에서 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기 실패 시
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 프레임을 창에 표시
        cv2.imshow("Camera Test", frame)

        # ESC 키로 종료
        if cv2.waitKey(1) == 27:
            print("Exiting camera test.")
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    test_camera()