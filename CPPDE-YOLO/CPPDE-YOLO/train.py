import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = YOLO(r'path')#
    model.train(data='path',
                cache=False,
                imgsz=224,
                epochs=200,
                single_cls=False, 
                batch=64,
                close_mosaic=0,
                workers=8,
                device='0',
                project='runs/train',
                name='exp',
                )