import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import torch
import logging
import torch.nn.functional as F


# 무게 관련 클래스 임포트
import get_weight as gw
import time

# weight = 0

class DepthVolumeCalculator:
    # def __init__(self, model_path, roi_points, cls_name_color, data_handler,food_processor):
    def __init__(self, model_path, roi_points, cls_name_color, food_processor):
        """
        초기화
        """
        
        # 무게 객체 생성
        self.GET_WEIGHT = gw.GetWeight()
        
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = None
        self.save_depth = None
        self.roi_points = roi_points
        self.cls_name_color = cls_name_color
        
        
        
        self.last_confirmed_obj_id = None   # 가장 최근 확정 객체 id 초기화
        self.candidate_objects = {}  # 객체 이름 및 프레임 수 기록
        # self.confirmed_objects = set()  # 확정된 객체
        self.confirmed_objects = {} # 확정된 객체
        # self.data_handler = data_handler  # 데이터 처리 클래스
        self.food_processor = food_processor  # 추가된 food_processor : 양 카테고리를 가져오기 위함
        self.model_name = os.path.basename(model_path) 
        
        

        # Tray BBox 초기화
        self.tray_bboxes = [
            (10, 10, 240, 280),   # 구역 1
            (230, 10, 400, 280),  # 구역 2
            (390, 10, 570, 280),  # 구역 3
            (560, 10, 770, 280),  # 구역 4
            (10, 270, 430, 630),  # 구역 5
            (420, 270, 800, 630)  # 구역 6
        ]

        # YOLO 모델 로드
        try:
            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"YOLO model '{os.path.basename(model_path)}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

    def initialize_camera(self):
        """카메라 초기화"""
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("Camera initialized.")
        
        

    def capture_frames(self):
        """정렬된 깊이 및 컬러 프레임 반환"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def preprocess_images(self, depth_frame, color_frame):
        """깊이 및 컬러 프레임 전처리"""
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.flip(depth_image, -1), cv2.flip(color_image, -1)

    def apply_roi(self, image):
        """ROI 적용"""
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        return image[y1:y2, x1:x2]
    
    def find_closest_tray_region(self, mask_indices):
        """
        PyTorch를 사용하여 GPU에서 구역(BBox) 판단.
        """
        mask_y, mask_x = mask_indices
        min_x = torch.min(mask_x)
        max_x = torch.max(mask_x)
        min_y = torch.min(mask_y)
        max_y = torch.max(mask_y)

        tray_bboxes = torch.tensor(self.tray_bboxes, device='cuda', dtype=torch.float32)
        inside_x = (min_x >= tray_bboxes[:, 0]) & (max_x <= tray_bboxes[:, 2])
        inside_y = (min_y >= tray_bboxes[:, 1]) & (max_y <= tray_bboxes[:, 3])
        inside = inside_x & inside_y

        # 포함된 첫 번째 구역 반환
        indices = torch.where(inside)[0]
        if len(indices) > 0:
            return indices[0].item() + 1

        # 포함되지 않은 경우 가장 가까운 구역 계산
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        tray_centers = torch.tensor(
            [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in self.tray_bboxes],
            device='cuda',
            dtype=torch.float32
        )
        distances = torch.sqrt((tray_centers[:, 0] - center_x) ** 2 + (tray_centers[:, 1] - center_y) ** 2)
        closest_index = torch.argmin(distances).item()
        return closest_index + 1  # 가장 가까운 구역 반환
    

    def calculate_volume_on_gpu(self, cropped_depth, depth_intrin, mask_indices):
        """
        GPU를 사용하여 객체의 부피를 계산
        """
        depth_tensor = torch.tensor(cropped_depth, device='cuda', dtype=torch.float32)
        mask_y, mask_x = mask_indices
        mask_tensor = (torch.as_tensor(mask_y, device='cuda'), torch.as_tensor(mask_x, device='cuda'))

        saved_depth_tensor = torch.tensor(self.save_depth, device='cuda', dtype=torch.float32)

        z_cm = depth_tensor[mask_tensor] / 10.0  # ROI 깊이 (cm)
        base_depth_cm = saved_depth_tensor[mask_tensor] / 10.0  # 기준 깊이 (cm)
        height_cm = torch.clamp(base_depth_cm - z_cm, min=0)

        pixel_area_cm2 = (z_cm ** 2) / (depth_intrin.fx * depth_intrin.fy)
        volume = torch.sum(height_cm * pixel_area_cm2).item()
        return volume
    

    def save_cropped_object_with_bbox(self, image, bbox, save_path, object_name):
        """
        지정된 BBox 영역을 크롭하고 저장. 파일 이름에 고유 번호를 추가.
        """
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]

        # 현재 폴더 내의 파일 갯수 확인
        existing_files = os.listdir(save_path)
        count = sum(1 for file in existing_files if file.startswith(object_name.replace(' ', '_')) and file.endswith('.jpg'))

        # 파일 이름 생성
        file_name = f"{object_name.replace(' ', '_')}_{count + 1}.jpg"  # 고유 번호 추가
        file_path = os.path.join(save_path, file_name)

        # 크롭된 이미지 저장
        cv2.imwrite(file_path, cropped)
        print(f"Saved cropped object to: {file_path}")

    def save_detected_objects(self, image, detected_objects, q_categories):
        """
        탐지된 객체를 저장. 폴더 이름은 모델 출력 ID를 그대로 사용.
        """
        save_base_path = os.path.join(os.getcwd(), "detected_objects")
        os.makedirs(save_base_path, exist_ok=True)

        print(f"[DEBUG] 저장되기전 디버그 detected_objects: {detected_objects}")
        print(f"[DEBUG] 저장되기전 디버그 q_categories: {q_categories}")

        for obj_id, obj_data in detected_objects.items():
            # 데이터 유효성 확인
            # if not isinstance(obj_data, tuple) or len(obj_data) != 5:
            if not isinstance(obj_data, dict):
                print(f"[ERROR] Invalid data format for {obj_id}: {obj_data}")
                continue

            # 데이터 추출
            obj_name = obj_data.get('obj_name', 'Unknown')
            region = obj_data.get('region', None)
            bbox = obj_data.get('bbox', None)
            volume = obj_data.get('volume', None)
            weight = obj_data.get('weight', None)

            # 카테고리 가져오기
            category = q_categories.get(obj_id, 'Unknown')
            if category == 'Unknown':
                print(f"[WARNING] No category found for obj_id {obj_id}")

            # obj_id/카테고리를 폴더 이름으로 사용
            save_path = os.path.join(save_base_path, obj_id, category)
            os.makedirs(save_path, exist_ok=True)

            # 크롭 및 저장
            self.save_cropped_object_with_bbox(image, bbox, save_path, obj_id)
            print(f"Saved {obj_id} in Region {region} with Volume: {volume:.1f} cm^3, weight : {weight}")

        print("All detected objects have been saved.")


    def visualize_results(self, blend_image, object_name, region, volume, color, mask_indices):
        """
        시각화: blend_image에 마스크를 적용하여 화면에 표시.
        """
        mask_y, mask_x = mask_indices

        # GPU 텐서를 CPU로 이동 후 NumPy 배열로 변환
        mask_y = mask_y.cpu().numpy()
        mask_x = mask_x.cpu().numpy()

        # 마스크 부분을 강조하여 blend_image 시각화
        blend_image[mask_y, mask_x] = (blend_image[mask_y, mask_x] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        # 텍스트 정보 표시
        text = f"{object_name}: {volume:.1f}cm^3 ({region})"
        cv2.putText(blend_image, text, (mask_x.min(), mask_y.min() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        

    def main_loop(self, customer_id):
        """메인 처리 루프"""
        self.initialize_camera()

        if os.path.exists('save_depth.npy'):
            self.save_depth = np.load('save_depth.npy')
            print("Loaded saved depth data.")
        else:
            self.save_depth = None
            print("No saved depth data found. Please save depth data.")


        try:
            while True:
                depth_frame, color_frame = self.capture_frames()
                if not depth_frame or not color_frame:
                    continue

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                depth_image, color_image = self.preprocess_images(depth_frame, color_frame)
                cropped_depth = self.apply_roi(depth_image)
                cropped_color = self.apply_roi(color_image)

                blend_image = cropped_color.copy()
                results = self.model(cropped_color)


                current_frame_objects = {}

                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data
                        original_size = (cropped_color.shape[0], cropped_color.shape[1])
                        resized_masks = F.interpolate(masks.unsqueeze(1), size=original_size, mode='bilinear', align_corners=False).squeeze(1)
                        classes = result.boxes.cls
                        # boxes = result.boxes.xyxy  # BBox 좌표 가져오기

                        for i, mask in enumerate(resized_masks):
                            obj_id = self.model.names[int(classes[i].item())]
                            obj_name, color, _ = self.cls_name_color.get(obj_id, ("Unknown", (255, 255, 255), 999))

                            mask_indices = torch.where(mask > 0.5)
                            mask_y, mask_x = mask_indices[0], mask_indices[1]

                            volume = self.calculate_volume_on_gpu(cropped_depth, depth_intrin, (mask_y, mask_x))
                            region = self.find_closest_tray_region((mask_y, mask_x))
                            
                            # 구역이 판단된 경우 해당 구역의 bbox 사용, 아닌 경우 마스크 영역으로 bbox 생성
                            if region is not None:
                                bbox = self.tray_bboxes[region - 1]
                            else:
                                print(f"Object {obj_name} not within any predefined region.")
                                bbox = (mask_x.min().cpu().item(), mask_y.min().cpu().item(), 
                                        mask_x.max().cpu().item(), mask_y.max().cpu().item())
                                
       
                            # print(f"gloabal weight = {weight}")
                            weight = None
                            current_frame_objects[obj_id] = (obj_name, region, volume, bbox, weight)
                            
                            self.visualize_results(blend_image, obj_name, region, volume, color, mask_indices)
                            
                            
                for obj_id, obj_data in current_frame_objects.items():
                    # obj_name, region, volume, bbox, weight = obj_data  # 데이터 unpacking
                    obj_name, region, volume, bbox, weight = obj_data  # 데이터 unpacking
                    # 후보 객체의 프레임 수 업데이트
                    if obj_id not in self.candidate_objects:
                    # if obj_name not in self.candidate_objects:
                        self.candidate_objects[obj_id] = 0
                    self.candidate_objects[obj_id] += 1

                    # 15프레임 이상 감지된 객체 처리
                    if self.candidate_objects[obj_id] >= 15 and obj_id not in self.confirmed_objects:
                        # obj_name, region, volume, bbox = obj_data  # 데이터 unpacking
                        print(f"New object confirmed: {obj_name}")
                        
                        
                        # 현재 객체 추가
                        self.confirmed_objects[obj_id] = {
                            "obj_name" : obj_name,
                            "region": region,
                            "bbox" : bbox,
                            "volume": volume,
                            
                        }
                        
                        print(f"Confirmed object: id : {obj_id}, name: {obj_name}, Volume: {volume:.1f} cm³, Region: {region}, weight : {weight}")
                        
                        self.last_confirmed_obj_id = obj_id
                        print(f"last_confirm_obj_id : {self.last_confirmed_obj_id}")
                        
                        gw.first_object_weight = round(float(gw.recent_weight_str),2)
                        time.sleep(2)

                        onair_recent_weight = round(float(gw.recent_weight_str),2)
                        print(f"실시간 무게값 = {onair_recent_weight}")
                        gw.obj_name = obj_name
                        object_weight = onair_recent_weight - gw.first_object_weight
                        
                        print(f"나는 {obj_name} : {object_weight}!!!")
                        
                        # 이전 객체에 weight 추가/ volume 업데이트
                        if self.last_confirmed_obj_id is not None:
                            if obj_id in current_frame_objects:
                                current_volume = current_frame_objects[obj_id][2]
                                self.confirmed_objects[obj_id]["volume"] = current_volume
                                print(f"부피업데이트 {obj_id}: {current_volume}")
                                
                            weight = object_weight
                            # print(f"다음객체들어와서 이제 무게 줄수있어 : {object_weight}")
                            self.confirmed_objects[self.last_confirmed_obj_id]["weight"] = object_weight
                            print(f"객체정보에 무게 추가: {self.confirmed_objects[self.last_confirmed_obj_id]}")
                            
                            
                    # 무게 변화 감지 및 업데이트
                    if gw.first_object_weight is not None:
                        current_weight = round(float(gw.recent_weight_str), 2)
                        weight_difference = abs(current_weight - gw.first_object_weight)   
                        if abs(weight_difference) >= 20:
                            print(f"[INFO] 무게 변화 감지: {obj_name}, 무게 증가량: {weight_difference:.2f} g")
                            
                            # 무게 변화 시점의 부피 저장
                            if obj_id in self.confirmed_objects:
                                # 현재 부피 업데이트
                                current_volume = current_frame_objects[obj_id][2]  # current_frame_objects에서 최신 부피 가져오기
                                self.confirmed_objects[obj_id]["volume"] = current_volume

                                # 무게 업데이트
                                self.confirmed_objects[obj_id]["weight"] = current_weight
                                print(f"업데이트된 객체: id: {obj_id}, name: {obj_name}, Volume: {current_volume:.1f} cm³, Updated Weight: {current_weight:.2f} g")
                            
                            # 기준 무게 재설정
                            gw.first_object_weight = current_weight

                        

                # 프레임에 감지되지 않은 객체는 후보 리스트에서 제거
                to_remove = [obj for obj in self.candidate_objects if obj not in current_frame_objects]
                for obj in to_remove:
                    del self.candidate_objects[obj]          
                        


                cv2.imshow("Results", blend_image)  # blend_image를 화면에 표시
                key = cv2.waitKey(1)
                if key == 27:  # ESC를 눌렀을 때
                    
                    # if len(self.candidate_objects) == len(self.confirmed_objects):
                        # plate_food_data를 생성하기 위해 데이터 수집
                    plate_food_data = []
                    
                    # 확정된객체(confirmed_objects)에서 아이디를 가져와야 카테고리 계산을 할수 있는데
                    # 이름을 가져와서 min_max_table에서 검색하려니 없지 ㅠㅠ 
                    for obj_id, obj_data in self.confirmed_objects.items():    
                        obj_name = obj_data['obj_name']
                        region = obj_data['region']
                        volume = obj_data['volume']
                        bbox = obj_data['bbox']
                        weight = obj_data.get('weight', 0)  # weight 값이 없으면 기본값 0
                        # obj_name, region, volume, bbox, weight= current_frame_objects[obj_id]
                        # obj_name, region, volume, bbox= current_frame_objects[obj_id]
                        # region, volume, bbox, weight= current_frame_objects[obj_id]
                        # weight 변수 추가하면 not enough values to unpack (expected 5, got 4)
                        
                        # weight = float(input(f"음식 ID {obj_id}에 대한 weight 입력: "))
                        plate_food_data.append((obj_id, weight, volume, region))
                        print(plate_food_data)
                    # FoodProcessor를 사용해 카테고리 처리 및 저장
                    if plate_food_data:
                        try:
                            print(f"나 plate_food_data야 : {plate_food_data}")
                            # FoodProcessor를 사용해 카테고리 처리(카테고리 가지러 갔다가 food_process단의 일들을 모두 해결)
                            # food_process단의 일이란? 칼로리 계산하고 csv파일 저장하는 일 
                            q_categories = self.food_processor.process_food_data(
                                plate_food_data,customer_id, self.food_processor.min_max_table
                            )
                            print(f"[DEBUG] FOOD_PROCESSOR에서 잘 가져왔는지 q_categories 확인: {q_categories}")
                            

                            # 이미지와 함께 저장
                            print("이미지 저장하러왔어 그다음에 에러나는지 잘 봐줘")
                            # self.save_detected_objects(image=cropped_color, detected_objects=current_frame_objects, q_categories = q_categories)
                            self.save_detected_objects(image=cropped_color, detected_objects=self.confirmed_objects, q_categories = q_categories)
                            print("Detected objects and categories saved successfully.")
                            self.candidate_objects.clear()
                        except Exception as e:
                            print(f"데이터 처리 중 오류 발생: {e}")
                    else:
                        print("Plate food data is empty, nothing to save.")
                    # else:
                    #     # 두 데이터의 갯수가 맞아야 저장됨 
                        # print(f"[DEBUG] Candidate objects: {self.candidate_objects}")
                        # print(f"[DEBUG] Confirmed objects: {self.confirmed_objects}")
                    #     print("Not all objects have been confirmed. Please wait.")
                    break
                
                elif key == ord('f'):  # 'f' 키를 눌러 초기화
                    self.candidate_objects.clear()
                    print("초기화되었습니다.")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


# class DataHandler:
#     def __init__(self):
#         self.detected_data = []  # 탐지된 데이터를 저장할 리스트
#         # self.current_weight = 0 

#     def process_detected_object(self, obj_id, volume, region, weight):
#         """탐지된 객체 데이터를 처리"""
#         detected_info = {"food_id": obj_id, "volume": volume, "region": region, "weight" : weight}
#         self.detected_data.append(detected_info)
#         print(f"DataHandler: Processed {detected_info}")

#     def get_detected_objects(self):
#         """탐지된 객체 데이터를 반환"""
#         return self.detected_data
    
#     def clear_detected_objects(self):
#         """탐지된 객체 데이터를 초기화"""
#         self.detected_data = []

# DataHandler 생성
# data_handler = DataHandler()

# if __name__ == "__main__":
#     logging.getLogger("ultralytics").setLevel(logging.WARNING)
#     CLS_NAME_COLOR = {
#         '01011001': ('Rice', (255, 0, 255), 0),
#         '04017001': ('Soybean Soup', (0, 255, 255), 1),
#         '06012004': ('Tteokgalbi', (0, 255, 0), 2),
#         '07014001': ('Egg Roll', (0, 0, 255), 3),
#         '11013007': ('Spinach Namul', (255, 255, 0), 4),
#         '12011008': ('Kimchi', (100, 100, 100), 5)
#     }
#     MODEL_PATH = os.path.join(os.getcwd(), 'model', "large_epoch300.pt")
#     ROI_POINTS = [(175, 50), (1055, 690)]
#     data_handler = DataHandler()
#     calculator = DepthVolumeCalculator(MODEL_PATH, ROI_POINTS, CLS_NAME_COLOR, data_handler)
#     calculator.main_loop()
