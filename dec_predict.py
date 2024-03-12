import argparse
from ultralytics import YOLO


def combine_tensor(tensor1, tensor2):
    result_list = []
    for i in range(len(tensor1)):
        label = int(tensor2[i])
        values = ' '.join([str(val.item()) for val in tensor1[i]])
        result_list.append(f"{label} {values}")
    return result_list


def get_bbox(img_path):
    # Load a model
    model = YOLO('best.pt')  # pretrained YOLOv8n model
    results = model(img_path)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        tensor1 = boxes.xywhn.cpu()
        tensor2 = boxes.cls.cpu()
    return combine_tensor(tensor1, tensor2)


def parse_opt():
    parser = argparse.ArgumentParser(description='Detection Module')
    parser.add_argument("--img_path", type=str,
                        default=r'20180824-13-43-21-401.jpg')
    opt = parser.parse_args()
    return opt


def main(opt):
    get_bbox(opt.img_path)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
