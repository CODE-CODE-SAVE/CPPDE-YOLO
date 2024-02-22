import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('H:\\zhaojiefeng\\yolov8\\yolov8\\runs\\train\\exp2\\weights\\best.pt')
    model.val(data='H:\\zhaojiefeng\\bookcoverdataset\\imagenet',
              split='val',
              imgsz=224,
              batch=128,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )