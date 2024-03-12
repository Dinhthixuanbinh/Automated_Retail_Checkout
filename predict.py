from PIL import Image
from tools.process_input import convert_box, rotate_cropped_img
from tools.process_output import model_pred, get_name_id, plot_bbox
from dec_predict import get_bbox
from tools.create_database import create_module
import argparse


def inferecne_img(img_path,
                  embedding_model,
                  id_class_dict,
                  database):
    img = Image.open(img_path)
    img_width, img_height = img.size
    bbox = get_bbox(img_path)
    predictions = []

    # Iterate through each line in the label file
    for iter, line in enumerate(bbox):
        # Extract class label and bounding box coordinates
        class_id, x_center, y_center, width, height = map(float, line.split())

        # Convert normalized coordinates to absolute coordinates
        left, top, right, bottom = convert_box(x_center, y_center,
                                               width, height,
                                               img_height, img_width)
        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        W, H = cropped_img.size

        if W > H:
            cropped_img = rotate_cropped_img(img_path,
                                             bbox, iter,
                                             state=False)
        name_id = get_name_id(dataframe='class_names_with_index.csv',
                              number_id=int(class_id))
        index_pred = model_pred(embedding_model, cropped_img,
                                name_id, database,
                                id_class_dict, state=True)
        predictions.append((f'{name_id}_{index_pred}',
                            (left, top, right, bottom)))

    return predictions


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict Image')
    parser.add_argument("--directory_path", type=str,
                        default=r'label_crop_top10')
    parser.add_argument("--img_path", type=str,
                        default=r'val_level_img\easy\20180824-13-43-21-401.jpg')
    parser.add_argument("--id_class_path", type=str,
                        default=r'map_id.csv')
    opt = parser.parse_args()
    return opt


def main(opt):
    embedd_model, database, id_class_dict = create_module(opt.directory_path,
                                                          opt.id_class_path)
    predictions = inferecne_img(opt.img_path, embedd_model,
                                id_class_dict, database)
    plot_bbox(opt.img_path, predictions)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
