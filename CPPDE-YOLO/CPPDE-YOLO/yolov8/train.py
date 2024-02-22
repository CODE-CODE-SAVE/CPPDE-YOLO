import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 设置使用编号为1的GPU
    model = YOLO(r'H:\zhaojiefeng\yolov8fuben\yolov8\ultralytics\cfg\models\v8\yolov8s-new-success.yaml')##yolov8s-new-success.yaml  yolov8-FasterNet.yaml

    #model = YOLO('H:/zhaojiefeng/yolov8/yolov8/runs/train/exp9/weights/last.pt')
    # model.load('H:\\zhaojiefeng\\yolov8\\yolov8s-cls.pt') # loading pretrain weights

    model.train(data='H:\\zhaojiefeng\\bookcoverdataset\\imagenet',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=224,
                epochs=200,
                single_cls=False,  # 是否是单类别检测
                batch=64,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD',     # using SGD
                resume="",         # 如过想续训就设置last.pt的地址
                amp=False,      # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )