from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':

    # Load a model
    model = YOLO(r"your/path/best.pt")  # load an official model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print("map50-95", metrics.box.map)  # map50-95
    print("map50", metrics.box.map50)  # map50
    print("map75", metrics.box.map75)  # map75
    print("box.maps", metrics.box.maps)  # a list contains map50-95 of each category