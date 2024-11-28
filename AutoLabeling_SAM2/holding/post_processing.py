import cv2
import numpy as np

def merge_grounded_sam_segments(image_path, txt_path):
    """
    Grounded-SAM의 segmentation 결과를 하나의 영역으로 병합합니다.

    Args:
        image_path: 이미지 파일 경로
        txt_path: Grounded-SAM segmentation 결과 파일 경로

    Returns:
        병합된 segmentation 마스크 이미지 (NumPy array) 또는 None (오류 발생 시)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        h, w = img.shape[:2]  # 이미지 높이, 너비

        mask = np.zeros((h, w), dtype=np.uint8)

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            coords = np.fromstring(line, dtype=float, sep=' ')
            for i in range(0, len(coords), 2):
                x = int(coords[i + 1] * w)
                y = int(coords[i] * h)

                # 좌표 유효성 검사 추가 (디버깅 4번)
                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = 255
                else:
                    print(f"Warning: 유효하지 않은 좌표: ({x}, {y}). 이미지 크기: ({w}, {h})")


        # 연결된 구성 요소 찾기
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # 디버깅 3번: labels 배열 크기, 최대값, nlabels 출력
        print("labels shape:", labels.shape)
        print("merged_mask shape:", mask.shape) # mask.shape 출력
        print("max(labels):", np.max(labels))
        print("nlabels:", nlabels)


        largest_label = 1
        largest_area = 0
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_label = i

        # 가장 큰 영역의 마스크 생성 (디버깅 1번 해결)
        merged_mask = np.zeros_like(labels, dtype=np.uint8)  # labels와 같은 크기와 타입
        merged_mask[labels == largest_label] = 255


        return merged_mask


    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error during processing: {e}")
        return None


# 예시 사용법:
image_path = "C:/Users/han/Desktop/validation/2nd_dataset(resized)/Post_processing_test/04_041_04011007_160273723933160.jpg"  # 이미지 파일 경로
txt_path = "C:/Users/han/Desktop/validation/2nd_dataset(resized)/Post_processing_test/04_041_04011007_160273723933160.txt"  # segmentation 결과 txt 파일 경로

merged_mask = merge_grounded_sam_segments(image_path, txt_path)



if merged_mask is not None:
    img = cv2.imread(image_path)
    result = cv2.bitwise_and(img, img, mask=merged_mask)
    cv2.imshow("Merged Mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("merged_mask.png", merged_mask)