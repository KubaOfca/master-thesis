import cv2
import numpy as np
from typing import Union
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os
import logging
import pandas as pd
from pathlib import Path

def load_image(path: str) -> Union[np.ndarray, None]:
    image = cv2.imread(path)
    logging.info(f"Image '{path}' loaded correctly!")
    return image

def print_tree(p: Path, last=True, header=''):
    elbow = "└──"
    pipe = "│  "
    tee = "├──"
    blank = "   "
    print(header + (elbow if last else tee) + p.name)
    if p.is_dir():
        children = list(p.iterdir())
        for i, c in enumerate(children):
            print_tree(c, header=header + (blank if last else pipe), last=i == len(children) - 1)

def show_raw_images(dir: str, n=10) -> None:
    random_images_names = random.sample(os.listdir(dir), n)
    logging.debug(f"Random images path: {random_images_names}")
    random_images = [load_image(os.path.join(dir, path)) for path in random_images_names]
    rows_len = cols_len = int(math.sqrt(n))
    fig, axes = plt.subplots(rows_len, cols_len, figsize=(10, 10))
    for ax, image, image_name in zip(axes.flatten(), random_images, random_images_names):
        ax.imshow(image)
        ax.set_title(image_name)
        ax.set_ylabel("Image height [px]")
        ax.set_ylim(image.shape[0], 0)
        ax.set_yticks(np.linspace(*ax.get_ylim(), 5))
        ax.set_xlabel("Image width [px]")
        ax.set_xlim(0, image.shape[1])
        ax.set_xticks(np.linspace(*ax.get_xlim(), 5))
    plt.tight_layout()
    plt.show()


def get_folder_size(dir):
    size=0
    for file in os.scandir(dir):
        size+=os.path.getsize(file)
    print("Dataset size:", size // (1024**2), "[MB]")


def pie_chart_from_list(data: list) -> None:

    counter = Counter(data)
    labels = list(counter.keys())
    values = list(counter.values())
    plt.pie(
        values,
        labels=labels,
        shadow=True,
        autopct='%1.1f%%',
        textprops={
            'size': 'medium',
            'fontweight': 'bold'
        },
    )
    plt.tight_layout()
    plt.show()


def transform_labels_to_yolo_format(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Transform various annotation format to suit YOLO annotation format.

    YOLO annotation format only accepts bounding boxes described as follows:
    - label: representation of class (int) 
    - x_center: x coordinates of the center of the box
    - y_center: y coordinates of the center of the box
    - width: width of the box
    - height: height of the box

    All these values need to be normalized by the image size.

    Args:
        annotation (pd.DataFrame): DataFrame containing annotations that are not in YOLO format.
        type (str, optional): Type of annotations that are not in YOLO format. TLBR - top left bottom right. Defaults to "TLBR".

    Returns:
        pd.DataFrame: DataFrame with YOLO annotation format
    """
        
    labels["x_center"] = (labels["x1"]
                              .add(labels["x2"])
                              .div(2)
                              .div(labels["img_width"]))
    labels["y_center"] = (labels["y1"]
                              .add(labels["y2"])
                              .div(2)
                              .div(labels["img_height"]))
    labels["width"] = (labels["x2"]
                              .sub(labels["x1"])
                              .div(labels["img_width"]))
    labels["height"] = (labels["y2"]
                              .sub(labels["y1"])
                              .div(labels["img_height"]))

    return labels[["img_name", "label", "x_center", "y_center", "width", "height"]]


def save_yolo_labels(labels, dir):
    for image_name, g_labels in tqdm(labels.groupby("img_name")):
        try:
            label_file_name = os.path.join(dir, f"{image_name}.txt")
            if not os.path.exists(label_file_name):
                g_labels.iloc[:, 1:].to_csv(label_file_name, header=False, index=False, mode='a', sep=" ")
            else:
                logging.info(f"{image_name} labels already exists")
        except Exception as e:
            logging.exception(e)
            logging.exception(f"Failed to save: {image_name}")


def convert_xywhn_to_x1y1x2y2(labels, image_width, image_height):
    df = labels.copy()
    df["x1"] = ((df.iloc[:, 1] - df.iloc[:, 3] / 2) * image_width).astype("int64")
    df["y1"] = ((df.iloc[:, 2] - df.iloc[:, 4] / 2) * image_height).astype("int64")
    df["x2"] = ((df.iloc[:, 1] + df.iloc[:, 3] / 2) * image_width).astype("int64")
    df["y2"] = ((df.iloc[:, 2] + df.iloc[:, 4] / 2) * image_height).astype("int64")
    df["label"] = df.iloc[:, 0]
    return df[["x1", "y1", "x2", "y2", "label"]]


def show_labeled_image(image_path: str, label_path: str, label_map: dict) -> None:
    image = load_image(image_path)
    label = pd.read_csv(label_path, sep= " ", header=None)
    label_convert = convert_xywhn_to_x1y1x2y2(label, image.shape[1], image.shape[0])
    box_color = (0, 255, 0)
    text_color = (0, 0, 255)
    box_thickness = 2
    text_thickness = 2
    text_font_size = 1
    for i, row in label_convert.iterrows():
        image = cv2.rectangle(image, (row["x1"], row["y1"]), (row["x2"], row["y2"]), box_color, box_thickness)
        cv2.putText(image, "class: " + label_map[str(row["label"])], (row["x1"], row["y1"] - 10), cv2.FONT_HERSHEY_SIMPLEX, text_font_size, text_color, text_thickness)
    plt.title(os.path.basename(image_path))
    plt.imshow(image)
    plt.show()