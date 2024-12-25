import pandas as pd
from datetime import datetime
import numpy as np
# from camera_copy_부피와무게추가 import DataHandler, DepthVolumeCalculator
from camera_copy_부피와무게추가 import DepthVolumeCalculator
import logging
from datetime import datetime, time

# 무게 관련 클래스 임포트
import get_weight as gw
# import time

# 안면인식 관련 클래스 임포트
from face import FaceRecognitionSystem


# 고객정보 담당 & 기초대사랑 & 한끼 권장 식사량 게산클래스
class CustomerManager:
    def __init__(self, user_csv_file_path):
        self.user_csv_file_path = user_csv_file_path

    # 고객정보를 CSV파일에서 가져오는 함수
    def get_customer_info(self, customer_id):   # 고객아이디와 고객정보파일경로를 매개변수로 가져옴
        try:
            # csv파일 읽기(연락처를 문자열로 변환)
            customer_data = pd.read_csv(self.user_csv_file_path, dtype={'phone': object})
            # 매개변수로 주어진 고객아이디로 고객정보 필터링
            customer_info = customer_data[customer_data['id'] == customer_id]
            if not customer_info.empty:
                # orient : 출력한 dict의 형태를 지정 / records : [ { 열 : 값 , 열 : 값 }, { 열 : 값, 열 : 값 } ]
                return customer_info.to_dict(orient='records')[0]
            else:
                return None
        except FileNotFoundError:
            print(f"CSV 파일 {self.user_csv_file_path}이(가) 존재하지 않습니다.")
            return None

    # 만나이 계산
    def calculate_age(self, birth_date):    # 8자리의 생년월일 
        
        today = datetime.now()
        birth_year, birth_month, birth_day = int(birth_date[:4]), int(birth_date[4:6]), int(birth_date[6:])
        # 현재 날짜와 비교하여 만 나이 계산
        age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
        return age
    
    # bmr(기초대사량) 게산 
    def calculate_bmr(self, weight, height, age, sex, exercise_score):
        # Mifflin-st-jeor 공식으로 BMR 계산
        bmr = (10 * weight) + (6.25 * height) - (5 * age)
        # 성별에 따라 BMR 조정
        if sex == 'F':
            bmr -= 161
        elif sex == 'M':
            bmr += 5
        else:
            raise ValueError("성별은 'M' 또는 'F'로 입력해야 합니다.")

        activity_multiplier = {
            'A': 1.9, 'B': 1.725, 'C': 1.55, 'D': 1.375, 'E': 1.2
        }
        # 활동량에 따라 BMR에 활동량 계수를 곱해줌
        # 운동량 정보가 없으면(이제 막 가입한 회원은 운동량 정보가 없을 수 있음) 기본값 1.55를 곱해주자
        return bmr * activity_multiplier.get(exercise_score, 1.55)


class FoodProcessor:
    
    # def __init__(self, food_data_path, real_time_csv_path, customer_diet_csv_path, data_handler, min_max_table):
    def __init__(self, food_data_path, real_time_csv_path, customer_diet_csv_path, min_max_table):
        self.food_data_path = food_data_path
        self.real_time_csv_path = real_time_csv_path
        self.customer_diet_csv_path = customer_diet_csv_path
        # self.data_handler = data_handler  # DataHandler 객체 추가
        self.min_max_table = min_max_table  # min_max_table 추가
        # print(f"FoodProcessor initialized with DataHandler: {self.data_handler}")

    # 음식 칼로리 계산함수
    def calculate_nutrient(self, base_weight, base_value, consumed_weight):
        return base_value * (consumed_weight / base_weight)

    # 양추정 기준 데이터 불러오기
    def load_min_max_table(file_path):
        try:
            # 양추정의 기준이 되어 줄 최소/최대값 테이블 읽어오기(quantity_min_max.csv)
            min_max_table = pd.read_csv(file_path, encoding='utf-8', dtype = {'food_id' : str})
            # 반환
            return min_max_table
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV 파일 {file_path}을(를) 찾을 수 없습니다.")
        
    # 특정 음식의 양 추정 구간 설정
    def calculate_q_ranges(self,food_id, min_max_table):
        # 음식 아이디로 최소/최대량 정보 필터링
        food_info = min_max_table[min_max_table['food_id'] == food_id]
        if food_info.empty:
            raise ValueError(f"음식 ID {food_id}에 대한 정보를 찾을 수 없습니다.")
        min_quantity = food_info['min'].values[0]
        max_quantity = food_info['max'].values[0]
        # Q1-Q5 구간 설정(linspace: start, stop, num) 최소값 최대값 사이를 5구간으로 나누기
        quantities = np.linspace(min_quantity, max_quantity, 5)
        # Q1-Q5 구간 dictionary로 return
        q_ranges = {f"Q{i+1}": quantities[i] for i in range(len(quantities))}
        print(f'이거 어디나와? {q_ranges}') # 구간이 잘 나뉘어졌는지 확인해봄
        return q_ranges


    # 양 추정 코드 정의(Q1-Q5)
    def determine_q_category(self, measured_weight, q_ranges):    # 저울로 측정된 무게값과 (위 함수에서 반환해주는)아이디별 양 추정 구간을 매개변수로 필요로함
        q_values = list(q_ranges.values())  # q_ranges가 딕셔너리 형태로 반환되므로 값만 가져와서 리스트로 변환
        for i in range(len(q_values) - 1): # 순회를 하면서 저울로 측정된 무게가 어느 구간에 속하는지 뒤져본다
            if q_values[i] <= measured_weight <= q_values[i + 1]:
                return f"Q{i+1}" 
        # 범위를 벗어나면 q5나 q1으로 반환하는데 아예 관련없는 값으로 반환하여 실시간 음식정보에 저장되지 않도록 수정해야할지 고민
        return "Q5" if measured_weight > q_values[-1] else "Q1" 
    
    # 실시간 음식정보 저장
    def save_to_csv(self, file_path, data):
        try:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, data], ignore_index=True)
        except FileNotFoundError:
            updated_df = data
        updated_df.to_csv(file_path, index=False)
        print(f"데이터가 {file_path}에 저장되었습니다.")

    # 고객 식단 상세 정보 저장
    def save_customer_diet_detail(self, customer_id, total_weight, total_nutrients):
        today_date = datetime.now().strftime('%Y%m%d')
        try:
            existing_data = pd.read_csv(self.customer_diet_csv_path)
            same_date_data = existing_data[
                (existing_data['customer_id'] == customer_id) &
                (existing_data['log_id'].str.startswith(f"{today_date}_{customer_id}"))
            ]
            # 같은 날짜의 같은 고객 데이터가 존재하는 경우 숫자를 붙여서 로그아이디 생성
            log_number = len(same_date_data) + 1
        except FileNotFoundError:
            log_number = 1

        log_id = f"{today_date}_{customer_id}_{log_number}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_data = {
            "log_id": [log_id],
            "customer_id": [customer_id],
            "total_weight": [total_weight],
            **{f"total_{key}": [round(value, 2)] for key, value in total_nutrients.items()},
            "timestamp": [timestamp]
        }
        self.save_to_csv(self.customer_diet_csv_path, pd.DataFrame(new_data))

    # 음식 데이터 처리
    # 1. 음식의 양을 누적시키고 
    # 2. 칼로리를 계산하여 누적시키고 
    # 3. 누적시키는 동안 각각의 객체의 정보를 realtime food_info에 저장시킨다.
    # 4. 모든음식이 배식되고 end 입력하면(실제 서비스에서는 배식완료버튼을 클릭하면) 고객의 한끼 식사에 대한 정보를 테이블에 저장한다.
    def process_food_data(self, plate_food_data, customer_id, min_max_table):

        food_data = pd.read_csv(self.food_data_path, dtype={'food_id': str})

        # 초기값 설정
        total_nutrients = {
            'calories': 0, 'carb': 0, 'fat': 0, 'protein': 0,
            'ca': 0, 'p': 0, 'k': 0, 'fe': 0, 'zn': 0
        }
        total_weight = 0
        real_time_food_info = []


        # 한글명으로 출력하기 위해 컬럼값 매핑
        nutrient_mapping = {
            'calories': '칼로리',
            'carb': '탄수화물',
            'fat': '지방',
            'protein': '단백질',
            'ca': '칼슘',
            'p': '인',
            'k': '칼륨',
            'fe': '철',
            'zn': '아연'
        }
        
        # 음식양 범주 초기화
        categories = {}
        
        print(f'이건 내가 왜 찍어봤지? 푸드 프로세스 클래스에 있어 plate_food_data : {plate_food_data}')
        # 한가지 음식에 대한 아이디,무게,부피값이 plate_food_data에 담겨있음
        for food_id, measured_weight, measured_volume, bbox in plate_food_data:
            print(f"[DEBUG] plate_food안에 있어!  food_id: {food_id}, measured_weight: {measured_weight}, measured_volume: {measured_volume}, bbox: {bbox}")
            food_info = food_data[food_data['food_id'] == food_id]
            if not food_info.empty:
                # print(f"[DEBUG] Found food_info: {food_info}")
                food_info_dict = food_info.to_dict(orient='records')[0]
                base_weight = food_info_dict['weight(g)']
                nutrients = {
                    'calories': food_info_dict['calories(kcal)'],
                    'carb': food_info_dict['carb(g)'],
                    'fat': food_info_dict['fat(g)'],
                    'protein': food_info_dict['protein(g)'],
                    'ca': food_info_dict['ca(mg)'],
                    'p': food_info_dict['p(mg)'],
                    'k': food_info_dict['k(mg)'],
                    'fe': food_info_dict['fe(mg)'],
                    'zn': food_info_dict['zn(mg)']
                }


                calculated_nutrients = {key: self.calculate_nutrient(base_weight, value, measured_weight)
                                        for key, value in nutrients.items()}
                
                try:
                    # 음식에 대한 양 카테고리를 나누고
                    q_ranges = self.calculate_q_ranges(food_id, min_max_table)
                    print(f"[DEBUG] 음식에 대한 카테고리 나눠볼게! food_id: {food_id}, q_ranges: {q_ranges}, measured_weight: {measured_weight}")
                    # 특정 양이 어떤 범주에 속하는지 정의하고
                    category = self.determine_q_category(measured_weight, q_ranges)
                    print(f"[DEBUG] 그래서 범주가 요기야! food_id: {food_id}, category: {category}")
                    categories[food_id] = category
                except ValueError as e:
                    category = "Unknown"
                    print(f"[ERROR] food_id: {food_id}, error: {e}")


                real_time_food_info.append({
                    "food_id": food_id,
                    "name": food_info_dict['name'],
                    "volume": round(measured_volume, 2),
                    "weight": round(measured_weight, 2),
                    **{key: round(value, 2) for key, value in calculated_nutrients.items()},
                    "category": category,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                for key in total_nutrients:
                    total_nutrients[key] += calculated_nutrients[key]
                total_weight += measured_weight
            else:
                print(f"food_id {food_id}에 해당하는 음식 정보를 찾을 수 없습니다.")

        # 실시간 음식정보 CSV 파일 저장 
        if real_time_food_info:
            self.save_to_csv(self.real_time_csv_path, pd.DataFrame(real_time_food_info))
            
        # 고객 식단 상세 CSV 파일 저장
        self.save_customer_diet_detail(customer_id, total_weight, total_nutrients)

        # 총 음식 누적량과 음식 영양소정보(칼로리 포함) 계산 및 출력해주는 단계
        print(f"총 누적량: {total_weight:.2f}g")
        for key, value in total_nutrients.items():
            unit = "kcal" if key == "calories" else "g" if key in ["carb", "fat", "protein"] else "mg"
            print(f"총 {nutrient_mapping[key]}: {value:.2f} {unit}")
        print(f'food_process 단계에서 카테고리반환값 확인 : {categories}')
        
        # 카메라 종료버튼을 눌렀을때 양 범주를 토대로 이미지를 저장하는데 
        # 범주를 전달해주는 방법으로 이 함수의 반환값인 카테고리를 넘겨주는 방법을 선택했는데 
        # 생성된 데이터를 저장하고 칼로리 계산하는 모든행위를 카메라에서 ESC를 눌렀을때 처리되는 방향으로 어영부영 해결됨 ㅎㅎ
        return categories


# 전체적인 서비스를 실행하는 클래스
class FoodService:
    # def __init__(self, customer_manager, food_processor, min_max_table, depth_calculator, data_handler,face_recognition_system):
    def __init__(self, customer_manager, food_processor, min_max_table, depth_calculator,face_recognition_system):
        self.customer_manager = customer_manager
        self.food_processor = food_processor
        self.min_max_table = min_max_table  
        self.depth_calculator = depth_calculator  # DepthVolumeCalculator 객체 추가
        # self.data_handler = data_handler # dataHandler 객체 추가
        self.face_recognition_system = face_recognition_system  # FaceRecognitionSystem 객체 추가

        
    # 특정 시간대에 식사를 완료했는지 확인하는 메서드
    def check_meal_record(self, customer_id):
        # 현재 시간대 가져오기
        current_time = datetime.now().time()
        
        # 시간대 정의
        breakfast_time = (time(7, 0, 0), time(9, 0, 0))
        lunch_time = (time(11, 0, 0), time(14, 0, 0))
        dinner_time = (time(17, 0, 0), time(20, 0, 0))
        
        # 현재 시간대에 해당하는 식사 시간 구하기
        if breakfast_time[0] <= current_time <= breakfast_time[1]:
            time_name = "아침"
            time_range = breakfast_time
        elif lunch_time[0] <= current_time <= lunch_time[1]:
            time_name = "점심"
            time_range = lunch_time
        elif dinner_time[0] <= current_time <= dinner_time[1]:
            time_name = "저녁"
            time_range = dinner_time
        else:
            return False  # 현재는 식사 시간대가 아님
        
        try:
            # CSV 파일 읽기
            diet_data = pd.read_csv('FOOD_DB/customer_diet_detail.csv')
            # 고객 ID로 필터링
            customer_records = diet_data[diet_data['customer_id'] == customer_id]
            
            if customer_records.empty:
                return False  # 고객 기록이 없으면 False 반환
            
            # 시간대 내 기록 확인
            for _, record in customer_records.iterrows():
                record_time = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S').time()
                if time_range[0] <= record_time <= time_range[1]:
                    print(f"{time_name}식사를 이미 하셨습니다.")
                    return True
            
            return False  # 시간대 내 기록이 없으면 False 반환
        except FileNotFoundError:
            print("customer_diet_detail.csv 파일을 찾을 수 없습니다.")
            return False
        


    # 서비스 실행 메서드(메인 비스무리 한데 메인이 너무 길어질까봐 따로 뺌)
    def run(self):
        # 탐지된 데이터 초기화
        # self.depth_calculator.data_handler.clear_detected_objects()
        # customer_id = input("고객 ID를 입력하세요: ")
        print("Starting face recognition...")
        model, label_mapping = self.face_recognition_system.create_unified_model()
        if model is not None:
            customer_id = self.face_recognition_system.face_recognizer(model, label_mapping)
        else:
            print("Face recognition model creation failed. Exiting.")
            return
        customer_info = self.customer_manager.get_customer_info(customer_id)

        # 식사 기록 확인
        if self.check_meal_record(customer_id):
            print("다음 식사시간에 뵙겠습니다.")
            return
        
        # 식사 기록이 없으면 이후 로직 실행
        print("식사 기록이 없습니다. 다음 작업을 진행합니다.")


        # 고객 정보를 화면에 뿌려주기 위한 빌드업 
        if customer_info:
            try:
                birth_date = str(customer_info['birth'])
                weight = customer_info['weight']
                height = customer_info['height']
                sex = customer_info['gender']
                exercise_score = customer_info['exercise']
                name = customer_info['name']
                age = self.customer_manager.calculate_age(birth_date)
                bmr_value = self.customer_manager.calculate_bmr(weight, height, age, sex, exercise_score)
                one_meal_value = bmr_value / 3
                
                # 고객정보 출력하기
                print(f"고객명: {name}")
                print(f"나이: {age}세")
                print(f"성별: {sex}")
                print(f"운동량: {exercise_score}")
                print(f"BMR: {bmr_value:.2f} kcal")
                print(f"한 끼 권장 식사량: {one_meal_value:.2f} kcal")

                # DepthVolumeCalculator로 실시간 음식 탐지 및 부피 계산
                # customer_id 매개변수가 들어간 이유 : foodprocessor 클래스에서 음식 양 카테고리를 생성하고 
                # 그것을 토대로(범주에 따라서) 이미지를 잘라서 저장해야하기 때문에 
                # 양 범주를 가져오기 위해서 카메라 메인 루프에 customer_id를 넣어줌(minmaxtable은 그냥 읽어옴)
                self.depth_calculator.main_loop(customer_id) 


            except Exception as e:
                print(f"오류 발생: {e}")
        else:
            print("해당하는 고객 ID를 찾을 수 없습니다.")


# 실행
if __name__ == "__main__":
    user_csv_file_path = 'FOOD_DB/user_info.csv'
    real_time_csv_path = 'FOOD_DB/real_time_food_info.csv'
    customer_diet_csv_path = 'FOOD_DB/customer_diet_detail.csv'
    q_min_max_path = 'FOOD_DB/quantity_min_max.csv'

    # 이 파일을 굳이 여기에서 열 필요가 있을까싶긴한데 괜히 꼬일까봐 못건드는중..
    try:
        min_max_table = pd.read_csv(q_min_max_path, dtype = {'food_id' : str})
    except FileNotFoundError:
        print(f"CSV 파일 {q_min_max_path}을(를) 찾을 수 없습니다.")
        exit()
        
    # 안면인식 객체생성 
    face_recognition_system = FaceRecognitionSystem()

    # 데이터핸들러 객체생성
    # data_handler = DataHandler()
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    model_path = 'model/large_epoch200.pt'  # YOLO 모델 경로
    roi_points = [(175, 50), (1055, 690)]  # ROI 설정
    
    # 마스킹 컬러 정의
    cls_name_color = {
    '01011001': ('Rice', (255, 0, 255), 0),  # Steamed Rice
    '04017001': ('Soybean Soup', (0, 255, 255), 1),  # Soybean Paste Stew
    '06012004': ('Tteokgalbi', (0, 255, 0), 2),  # Grilled Short Rib Patties (Tteokgalbi)
    '07014001': ('Egg Roll', (0, 0, 255), 3),  # Rolled Omelette
    '11013007': ('Spinach Namul', (255, 255, 0), 4),  # Spinach Namul
    '12011008': ('Kimchi', (100, 100, 100), 5),  # Napa Cabbage Kimchi
    '01012006': ('Black Rice', (255, 0, 255), 6),  # Black Rice
    '04011005': ('Seaweed Soup', (0, 255, 255), 7),  # Seaweed Soup
    '04011007': ('Beef Stew', (0, 255, 255), 8),  # Beef Radish Soup
    '06012008': ('Beef Bulgogi', (0, 255, 0), 9),  # Beef Bulgogi
    '08011003': ('Stir-fried Anchovies', (0, 0, 255), 10),  # Stir-fried Anchovies
    '10012001': ('Chicken Gangjeong', (0, 0, 255), 11),  # Sweet and Spicy Fried Chicken
    '11013002': ('Fernbrake Namul', (255, 255, 0), 12),  # Fernbrake Namul
    '12011003': ('Radish Kimchi', (100, 100, 100), 13),  # Radish Kimchi (Kkakdugi)
    '01012002': ('Bean Rice', (255, 0, 255), 14),  # Soybean Rice
    '04011011': ('Fish Cake Soup', (0, 255, 255), 15),  # Fish Cake Soup
    '07013003': ('Kimchi Pancake', (0, 0, 255), 16),  # Kimchi Pancake
    '11013010': ('Bean Sprouts Namul', (255, 255, 0), 17),  # Bean Sprout Namul
    '03011011': ('Pumpkin Soup', (255, 0, 255), 18),  # Pumpkin Porridge
    '08012001': ('Stir-fried Potatoes', (255, 255, 0), 19)  # Stir-fried Potatoes
    }
    
    # 라즈베리파이 서버 연결
    # GET_WEIGHT = gw.GetWeight()
    


    # 객체생성
    food_processor = FoodProcessor(
        'FOOD_DB/food_project_food_info.csv', 
        real_time_csv_path, 
        customer_diet_csv_path, 
        # data_handler,
         min_max_table)
    
    # 객체생성
    depth_calculator = DepthVolumeCalculator(
        model_path, 
        roi_points, 
        cls_name_color, 
        # data_handler, 
        food_processor,
        )

    # 객체생성
    customer_manager = CustomerManager(user_csv_file_path)
    
    
    # FoodService 객체생성 및 실행
    # service = FoodService(customer_manager, food_processor, min_max_table, depth_calculator, data_handler,face_recognition_system)
    service = FoodService(customer_manager, food_processor, min_max_table, depth_calculator,face_recognition_system)
    service.run()
