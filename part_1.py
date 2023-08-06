import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any, Sequence
from scipy.ndimage import maximum_filter
import os


DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'
# The label we want to look for in the polygons json file
TFL_LABEL = ['traffic light']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def draw_bounding_boxes(image: np.ndarray, red_x: List[int], red_y: List[int], green_x: List[int],
                        green_y: List[int]) -> np.ndarray:
    """
    Draw bounding boxes around the red and green traffic lights in the image
    :param image: The image to draw the bounding boxes on
    :param red_x: The x coordinates of the red traffic lights
    :param red_y: The y coordinates of the red traffic lights
    :param green_x: The x coordinates of the green traffic lights
    :param green_y: The y coordinates of the green traffic lights
    :return: The image with the bounding boxes drawn on it
    """
    width, height = 30, 60

    # Iterate over the red and green coordinates, drawing rectangles of the fixed size around them
    for x, y in zip(red_x, red_y):
        top_left = (x - width // 2, y - height // 5)
        bottom_right = (x + width // 2, y + height // 2)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Red color for red light

    for x, y in zip(green_x, green_y):
        top_left = (x - width // 2, y - height // 2)
        bottom_right = (x + width // 2, y + height // 5)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green color for green light
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

    color_x, color_y = filter_close_points(color_x, color_y, 50)

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


# helper function for filtering too close points - it put one point instead of more than one
# calculates the average point for each cluster of close points, providing a more central representation of the objects.
def filter_close_points(x_coords: List[int], y_coords: List[int], min_distance: int) -> Tuple[List[int], List[int]]:
    filtered_x, filtered_y = [], []
    points = sorted(zip(x_coords, y_coords))
    if not points:
        return filtered_x, filtered_y
    prev_x, prev_y = points[0]
    count = 1
    sum_x, sum_y = prev_x, prev_y

    for x, y in points[1:]:
        if np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2) < min_distance:
            sum_x += x
            sum_y += y
            count += 1
        else:
            filtered_x.append(int(sum_x / count))
            filtered_y.append(int(sum_y / count))
            prev_x, prev_y = x, y
            sum_x, sum_y = prev_x, prev_y
            count = 1

    if count > 0:
        filtered_x.append(int(sum_x / count))
        filtered_y.append(int(sum_y / count))

    return filtered_x, filtered_y


def crop_image(image: np.ndarray, x: int, y: int, color: str) -> np.ndarray:
    """
    Crop the image around the given coordinates
    :param image: The image to crop
    :param x: The x coordinate
    :param y: The y coordinate
    :param color: The color of the traffic light
    :return: The cropped image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width, height = 30, 60

    if color == 'ro':
        top_left = (max(0, x - width // 2), max(0, y - height // 5))
        bottom_right = (min(image.shape[1], x + width // 2), min(image.shape[0], y + height // 2))
    elif color == 'go':
        top_left = (max(0, x - width // 2), max(0, y - height // 2))
        bottom_right = (min(image.shape[1], x + width // 2), min(image.shape[0], y + height // 5))

    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_image


def save_cropped_images(crop: np.ndarray, index: int, color: str) -> np.ndarray:
    """
    Save the cropped image
    :param crop: The cropped image
    :param index: The index of the image
    :param color: The color of the traffic light
    :return: The cropped image
    """
    if not os.path.exists('cropped_images'):
        os.makedirs('cropped_images')

    cv2.imwrite(f'cropped_images/{color}{index}.png', crop)


def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'b'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
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


def crop_and_save(image, xs, ys, color):
    """
    Crop the image around the given coordinates and save it
    :param image: The image to crop
    :param xs: The x coordinates
    :param ys: The y coordinates
    :param color: The color of the traffic light
    :return: The cropped images
    """
    crops = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        crop = crop_image(image, x, y, color)
        save_cropped_images(crop, i, color)
        crops.append(crop)
    return crops


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
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]
    show_image_and_gt(c_image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
    bounded_image = draw_bounding_boxes(c_image, red_x, red_y, green_x, green_y)
    plt.imshow(bounded_image)

    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)

    crops_r = crop_and_save(c_image, red_x, red_y, 'ro')
    crops_g = crop_and_save(c_image, green_x, green_y, 'go')
    plot_cropped_images(c_image, crops_r + crops_g)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.
    :param argv: In case you want to programmatically run this.
    """
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    # If you entered a custom dir to run from or the default dir exist in your project then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))
        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)
    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
