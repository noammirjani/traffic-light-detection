import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any, Sequence
import os
from sklearn.cluster import DBSCAN
from pandas import DataFrame
from shapely.geometry import Polygon, box

# Constants from your provided code
SEQ: str = 'seq'
IS_TRUE: str = 'is_true'
IGNOR: str = 'is_ignore'
CROP_PATH: str = 'path'
X0: str = 'x0'
X1: str = 'x1'
Y0: str = 'y0'
Y1: str = 'y1'
COL: str = 'col'
SEQ_IMAG: str = 'seq_imag'
GTIM_PATH: str = 'gtim_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COL]
BASE_SNC_DIR: Path = Path.cwd()
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
CROP_DIR: Path = DATA_DIR / 'crops'
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'
CROP_CSV_NAME: str = 'crop_results.csv'

# Existing code constants
DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'
TFL_LABEL = ['traffic light']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def make_crop(image: np.ndarray, x: int, y: int, color: str, **kwargs) -> Tuple[int, int, int, int, np.ndarray]:

    # Adjust the crop area depending on the color
    if color == 'red':
        x0, x1 = max(0, x - 35), min(image.shape[1], x + 35)
        y0, y1 = max(0, y - 80), min(image.shape[0], y + 160)
    else: # Assuming green or other colors
        x0, x1 = max(0, x - 35), min(image.shape[1], x + 35)
        y0, y1 = max(0, y - 160), min(image.shape[0], y + 80)

    cropped_image = image[y0:y1, x0:x1]
    resized_image = cv2.resize(cropped_image, (70, 240))
    return x0, x1, y0, y1, resized_image


def check_crop(image_json_path: str, x0: int, y0: int, x1: int, y1: int, color: str) -> Tuple[bool, bool]:
    """
    Check if the crop is valid
    :param image_json_path: The path to the json file
    :param x0: The x0 coordinate
    :param y0: The y0 coordinate
    :param x1: The x1 coordinate
    :param y1: The y1 coordinate
    :param color: The color of the traffic light
    :return: A tuple of two booleans, the first indicates if the crop is valid,
             the second indicates if the crop should be ignored
    """
    image_json = json.load(Path(image_json_path).open())
    tfls = [image_object for image_object in image_json['objects'] if image_object['label'] == "traffic light"]

    tfl_count = 0
    crop_img = box(x0, y0, x1, y1)

    for tfl in tfls:
        poly = Polygon(tfl['polygon'])
        if crop_img.contains(poly):
            tfl_count += 1
        elif crop_img.intersects(poly):
            return False, True  # half traffic light, ignore

    if tfl_count == 1:
        return True, False  # exactly one traffic light, don't ignore

    if tfl_count > 1:
        return False, True  # more than one traffic light, ignore

    return False, False  # no traffic light, don't ignore


def save_for_part_2(crops_df: DataFrame):
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()
    crops_sorted: DataFrame = crops_df.sort_values(by=SEQ)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def create_crops(c_image, df: DataFrame, image_json_path: Optional[str] = None) -> DataFrame:
    if not CROP_DIR.exists():
        CROP_DIR.mkdir(parents=True)  # Ensure the directory is created if it doesn't exist
    result_df = DataFrame(columns=CROP_RESULT)
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]
        x0, x1, y0, y1, crop = make_crop(c_image, row[X], row[Y], row[COLOR])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1

        # Extract the image's name from its path
        image_name = os.path.basename(row[GTIM_PATH]).split('.')[0]
        # Create a unique filename for the cropped image
        crop_filename = f"{image_name}_{row[COLOR]}_{index}.png"
        # Construct the full path for the cropped image
        crop_path = CROP_DIR / crop_filename

        result_template[CROP_PATH] = str(crop_path)
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        result_template[IS_TRUE], result_template[IGNOR] = check_crop(image_json_path, x0, y0, x1, y1, row[COLOR])
        result_df = result_df._append(result_template, ignore_index=True)  # Corrected the _append to append
    save_for_part_2(result_df)
    return result_df


def draw_bounding_boxes(image: np.ndarray, red_x: List[int], red_y: List[int], green_x: List[int],
                        green_y: List[int]) -> np.ndarray:
    for x, y in zip(red_x, red_y):
        top_left = (max(0, x - 35), max(0, y - 80))
        bottom_right = (min(image.shape[1], x + 35), min(image.shape[0], y + 160))
        image = cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    for x, y in zip(green_x, green_y):
        top_left = (max(0, x - 35), max(0, y - 160)) # Consistent with red
        bottom_right = (min(image.shape[1], x + 35), min(image.shape[0], y + 80)) # Consistent with red
        image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image


def find_ftl_single_color(c_image: np.ndarray, hsv_image: np.ndarray, kernel: np.ndarray, lower_color: np.array,
                          upper_color: np.array, threshold_val, num) \
        -> tuple[Union[list[int], list[Any]], Union[list[int], list[Any]], Sequence[Any]]:
    """
    Find the red or green traffic lights in the image
    :param c_image: The image in color
    :param hsv_image: The image in HSV
    :param kernel: The kernel to use for morphological operations
    :param lower_color: The lower threshold for the color
    :param upper_color: The upper threshold for the color
    :param threshold_val: The threshold value to use
    :param num: The color channel to use
    :return: The x and y coordinates of the traffic lights, and the contours of the traffic lights
    """
    # Create masks for red and green colors using the specified thresholds
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    # remove noise
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    # get the color channel
    color_channel = c_image[:, :, num]
    # apply the mask on the color channel
    filtered_channel = cv2.bitwise_and(color_mask, color_channel)

    # apply threshold
    color_threshold = (filtered_channel > threshold_val).astype(np.uint8)
    color_contours, _ = cv2.findContours(color_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the center of mass of the red and green areas in the image
    color_x = [int(np.mean(contour[:, :, 0])) for contour in color_contours]
    color_y = [int(np.mean(contour[:, :, 1])) for contour in color_contours]

    color_x, color_y = filter_close_points(color_x, color_y)
    return color_x, color_y, color_contours


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Find the red and green traffic lights in the image
    :param c_image: The image in color
    :param kwargs: The arguments to pass to the find_ftl_single_color function
    :return: The x and y coordinates of the red and green traffic lights
    """
    kernel = (1 / 13) * np.array([[-1, -1, -1, -1, -1],
                                  [-1, -1, 4, -1, -1],
                                  [-1, 4, 4, 4, -1],
                                  [-1, -1, 4, -1, -1],
                                  [-1, -1, -1, -1, -1]])

    # Convert the image to the HSV color space for better color detection
    hsv_image = cv2.cvtColor(c_image, cv2.COLOR_RGB2HSV)

    # Define the lower and upper thresholds for red and green colors in HSV space
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([100, 255, 255])

    # Find the red and green areas in the image
    red_x, red_y, red_contours = find_ftl_single_color(c_image, hsv_image, kernel, lower_red, upper_red, 100, 2)
    green_x, green_y, green_contours = find_ftl_single_color(c_image, hsv_image, kernel, lower_green, upper_green, 180, 1)

    return red_x, red_y, green_x, green_y


def filter_close_points(x_coords: List[int], y_coords: List[int]) -> Tuple[List[int], List[int]]:
    filtered_x, filtered_y = [], []
    points = np.array(list(zip(x_coords, y_coords)))

    if len(points) == 0:
        return filtered_x, filtered_y

    clustering = DBSCAN(eps=50, min_samples=1).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)

    for label in unique_labels:
        cluster_points = points[labels == label]
        center_x = np.mean(cluster_points[:, 0])
        center_y = np.mean(cluster_points[:, 1])
        filtered_x.append(int(center_x))
        filtered_y.append(int(center_y))

    return filtered_x, filtered_y


def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    plt.figure(fig_num).clf()
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            poly: np.array = np.array(image_object['polygon'])
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'b'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            plt.legend()


def plot_cropped_images(original_image, cropped_images):
    """
    Plot the original image with the bounding boxes and the cropped images
    :param original_image: The original image
    :param cropped_images: The cropped images
    """
    plt.figure(figsize=(20, 10))
    # Plot the original image with bounding boxes
    plt.subplot(2, 1, 1)
    plt.imshow(original_image)
    plt.title("Original Image with Bounding Boxes")

    # Plot the cropped images
    n = len(cropped_images)
    for i, img in enumerate(cropped_images):
        plt.subplot(2, n, n + i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.show()


# def crop_and_save(image, xs, ys, color):
#     """
#     Crop the image around the given coordinates and save it
#     :param image: The image to crop
#     :param xs: The x coordinates
#     :param ys: The y coordinates
#     :param color: The color of the traffic light
#     :return: The cropped images
#     """
#     crops = []
#     for i, (x, y) in enumerate(zip(xs, ys)):
#         crop = crop_image(image, x, y, color)
#         save_cropped_images(crop, i, color)
#         crops.append(crop)
#     return crops


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None):
    """
    Run the attention code.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)
    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open(encoding="utf-8"))
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]
    show_image_and_gt(c_image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
    bounded_image = draw_bounding_boxes(c_image, red_x, red_y, green_x, green_y)
    plt.imshow(bounded_image)

    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)

    # crops_r = crop_and_save(c_image, red_x, red_y, 'ro')
    # crops_g = crop_and_save(c_image, green_x, green_y, 'go')
    # plot_cropped_images(c_image, crops_r + crops_g)


def crop_image(image: np.ndarray, x: int, y: int, color: str) -> np.ndarray:
    """
    Crop the image around the given coordinates based on the traffic light's color.
    :param image: The image to crop.
    :param x: The x-coordinate of the center of the traffic light.
    :param y: The y-coordinate of the center of the traffic light.
    :param color: The color of the traffic light ('ro' for red, 'go' for green).
    :return: The cropped image.
    """
    if color == 'ro':  # Red traffic light
        x0, x1 = max(0, x - 35), min(image.shape[1], x + 35)
        y0, y1 = max(0, y - 80), min(image.shape[0], y + 160)
    else:  # 'go' - Green traffic light
        x0, x1 = max(0, x - 35), min(image.shape[1], x + 35)
        y0, y1 = max(0, y - 160), min(image.shape[0], y + 80)

    cropped_img = image[y0:y1, x0:x1]
    return cropped_img

def integrated_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None):
    image: Image = Image.open(image_path)
    c_image: np.ndarray = np.array(image)
    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
    bounded_image = draw_bounding_boxes(c_image, red_x, red_y, green_x, green_y)
    plt.imshow(bounded_image)
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)

    data = []
    for x, y in zip(red_x, red_y):
        data.append([image_path, x, y, 0.5, "red"])
    for x, y in zip(green_x, green_y):
        data.append([image_path, x, y, 0.5, "green"])

    df = pd.DataFrame(data, columns=["path", "x", "y", "zoom", "color"])
    df.to_csv(DATA_DIR / ATTENTION_PATH / "traffic_lights_details.csv", index=False)

    df_data = {
        SEQ_IMAG: [i for i in range(len(red_x + green_x))],
        X: red_x + green_x,
        Y: red_y + green_y,
        COLOR: ["red"] * len(red_x) + ["green"] * len(green_x),
        GTIM_PATH: [image_path for _ in range(len(red_x + green_x))]
    }
    df = DataFrame(df_data)
    create_crops(c_image, df, image_json_path)

    crops_r = [crop_image(c_image, x, y, 'ro') for x, y in zip(red_x, red_y)]
    crops_g = [crop_image(c_image, x, y, 'go') for x, y in zip(green_x, green_y)]
    plot_cropped_images(c_image, crops_r + crops_g)

# Main function
def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))
        for image in file_list:
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            integrated_find_tfl_lights(image_path, image_json_path)
    elif args.image and args.json:
        integrated_find_tfl_lights(args.image, args.json)
    elif args.image:
        integrated_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == "__main__":
    main()