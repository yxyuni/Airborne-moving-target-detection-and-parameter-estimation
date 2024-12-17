from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = YOLO('my_yolo/ours.yaml')
    model.train(data='your/path/data.yaml', epochs=500, imgsz=256, batch=8)
    result = model.val()
