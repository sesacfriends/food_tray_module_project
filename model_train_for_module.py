import os
import torch
from ultralytics import YOLO

def setEnvironment(osEnv):
    print("=== 환경 설정")
    device = torch.device('cpu')
    env = osEnv.lower()
    print(f"env = {env}")
    # windows, linux
    if (env == "windows") or (env == "linux"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # cuDNN 비활성화 -> gpu가 nvidia 1660일 때
        # torch.backends.cudnn.enabled = False
        print(f"using device : {device}")
    # mac
    elif env == "mac":
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f'Using device : {device}')
    else:
        device = torch.device('cpu')
    
    return device

def getModel(version='n'):
    print("--- 모델 준비")
    
    if version == 'n':
        model = YOLO("yolo11n-seg.pt")
    elif version == 's':
        model = YOLO("yolo11s-seg.pt")
    elif version == 'm':
        model = YOLO("yolo11m-seg.pt")
    elif version == 'l':
        model = YOLO("yolo11l-seg.pt")
    elif version == 'x':
        model = YOLO("yolo11x-seg.pt")
    
    print("--- 모델 준비 완료")
    return model

def trainModel(device, model, ypath):
    print("모델 학습")
    data_path = ypath
    model.train(data=data_path, device=device, epochs=50)
    

if __name__ == "__main__":
    print("모델 학습 준비과정 시작")
    
    # 운영체제 선택 - windows, linux, mac
    device = setEnvironment('Mac')
    print(f'device = {device}')
    
    # 모델 버전 선택 - n / s / m / l /  x
    model = getModel('n')
    # print(f'model = {model}')
    
    # yaml 파일 경로 수정
    ypath = '/Users/wonkyoung/food-project/data/data.yaml'
    trainModel(device, model, ypath)