import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import ndimage
from typing import List, Optional, Union, Dict, Tuple
from scipy.ndimage import maximum_filter

DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'
# The label we want to look for in the polygons json file
TFL_LABEL = ['traffic light']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def find_ftl_single_color(c_image: np.ndarray, hsv_image: np.ndarray, kernel: np.ndarray, lower_color: np.array,
                          upper_color: np.array, color_threshold) \
        -> Tuple[List[int], List[int]]:
    # Create masks for red and green colors using the specified thresholds
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    color_channel = c_image[:, :, 1]
    color_mask_channel = (color_channel > color_threshold).astype(np.uint8)
    color_mask = cv2.bitwise_and(color_mask, color_mask_channel)
    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color_x = [int(np.mean(contour[:, :, 0])) for contour in color_contours] if color_contours else []
    color_y = [int(np.mean(contour[:, :, 1])) for contour in color_contours] if color_contours else []
    return color_x, color_y


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> Tuple[List[int], List[int], List[int], List[int]]:
    c_image = ndimage.maximum_filter(c_image, 0.5)
    blur = cv2.GaussianBlur(c_image, (5, 5), 0.6)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]]) / 9

    # Convert the image to the HSV color space for better color detection
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    # Define the lower and upper thresholds for red and green colors in HSV space
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([100, 255, 255])

    # higher mean less points detected
    red_color_threshold = 65
    green_color_threshold = 145

    # Find the red and green areas in the image
    red_x, red_y = find_ftl_single_color(c_image, hsv_image, kernel, lower_red, upper_red, red_color_threshold)
    green_x, green_y = find_ftl_single_color(c_image, hsv_image, kernel, lower_green, upper_green,
                                             green_color_threshold)
    # Filter points that are too close, because we want one point for each object
    min_distance = 50
    red_x, red_y = filter_close_points(red_x, red_y, min_distance)
    green_x, green_y = filter_close_points(green_x, green_y, min_distance)

    #=-------------------------------------------------
    c_image_copy = c_image.copy()
    # Pass points as lists of tuples
    rectangles_drawing(c_image_copy, list(zip(red_x, red_y)), list(zip(green_x, green_y)))

    # Show the image with rectangles
    plt.imshow(c_image_copy)
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)

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
    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


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


# -----------------------------------------------------------------------------------------------------------

# we dont need (NMS) non max suppression because we already have one point on one object
# the NMS resposibale for filter out duplicate or overlapping bounding boxes (rectangles)

class Rectangle:
    def __init__(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

def rectangles_drawing(image: np.ndarray, red_points: List[tuple], green_points: List[tuple]):
    """
    Draw rectangles around points with a specified width and height.

    :param image: The input image as a numpy ndarray.
    :param red_points: A list of tuples containing the (x, y) coordinates of red points.
    :param green_points: A list of tuples containing the (x, y) coordinates of green points.
    """
    RED_COLOR = (255, 0, 0)
    GREEN_COLOR = (0, 255, 0)
    RECTANGLE_WIDTH = 30
    RED_RECTANGLE_HEIGHT = 80
    GREEN_RECTANGLE_HEIGHT = 80

    red_rectangles = [Rectangle(x - RECTANGLE_WIDTH//2, y - RED_RECTANGLE_HEIGHT//2,
                                x + RECTANGLE_WIDTH//2, y + RED_RECTANGLE_HEIGHT//2)
                      for x, y in red_points]

    green_rectangles = [Rectangle(x - RECTANGLE_WIDTH//2, y - GREEN_RECTANGLE_HEIGHT//2,
                                  x + RECTANGLE_WIDTH//2, y + GREEN_RECTANGLE_HEIGHT//2)
                        for x, y in green_points]

    for rect in red_rectangles:
        cv2.rectangle(image, (rect.top_left_x, rect.top_left_y), (rect.bottom_right_x, rect.bottom_right_y), RED_COLOR, 1)
        # Add the title above the rectangle
        cv2.putText(image, "Red Traffic", (rect.top_left_x, rect.top_left_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_COLOR, 1, cv2.LINE_AA)

    for rect in green_rectangles:
        cv2.rectangle(image, (rect.top_left_x, rect.top_left_y), (rect.bottom_right_x, rect.bottom_right_y), GREEN_COLOR, 1)
        # Add the title above the rectangle
        cv2.putText(image, "Green Traffic", (rect.top_left_x, rect.top_left_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR, 1, cv2.LINE_AA)





# -----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
