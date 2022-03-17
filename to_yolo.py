import numpy as np
from loguru import logger
from pathlib import Path
import pandas as pd
import cv2
import shutil
import os
import argparse


def convert2yolo(img, ann):
    df = pd.read_csv(ann, header=None, usecols=[0, 1, 2])
    df = df.iloc[1:].astype(float)  # ignore the header # X, Y, Radius
    vals = df.to_numpy()
    height, width = cv2.imread(str(img)).shape[:-1]

    yolo = []
    for val in vals:

        center_x = val[0]/width
        center_y = val[1]/height
        w_ratio = (min(val[0]+val[2], width) - max(val[0]-val[2], 0))/width * 0.9
        h_ratio = (min(val[1]+val[2], height) - max(val[1]-val[2], 0))/height * 0.9

        yolo.append([center_x, center_y, w_ratio, h_ratio])

    return np.array(yolo)

def generate(type, yolo, txt_path, CLASS):

    output = np.c_[np.ones(len(yolo))*CLASS.index(type), yolo]
    np.savetxt(txt_path, output.astype(np.float16), delimiter= " ", fmt='%i %1.4f %1.4f %1.4f %1.4f')

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes to check the annotation.")

    parser.add_argument("--input",
                        default="/media/hexaburbach/onetb/yolo_data/qrcode-datasets/datasets",
                        help="Location of label directory.")

    parser.add_argument("--image",
                        default='images',
                        help="Location of raw iamge directory to load for checking.")

    parser.add_argument("--ann",
                        default='anns',
                        help="Location of annotation directory to load for checking.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input = args.input
    CLASS = ["Person", "plants", "laptop", "cell phone", "QRcode"]
    target = "QRcode"
    save_im = args.image
    save_ann = args.ann

    """ Load Images and Annotations and sort. """
    anns = sorted(Path(input).glob("**/*.csv"), key=lambda path: str(path))
    img_ext = (".jpg", ".JPG")
    imgs = sorted(filter(lambda path: path.suffix in img_ext, Path(
        input).glob("**/*")), key=lambda path: str(path))

    logger.info(f"{len(imgs)} images will be processed.")
    assert len(imgs) == len(anns), "images and annotations are not matched!"

    for img, ann in zip(imgs, anns):
        yolo = convert2yolo(img, ann)
        ann_name = os.path.join( save_ann, img.parts[-2]+'_'+img.stem+'.txt' )
        shutil.copy( str(img), os.path.join(save_im, img.parts[-2]+'_'+img.name) )
        generate(target, yolo, ann_name, CLASS)

    logger.info("Process done!")
