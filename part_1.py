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
def find_ftl_single_color(c_image: np.ndarray, hsv_image: np.ndarray, kernel: np.ndarray, lower_color: np.array, upper_color: np.array,color_threshold)\
        -> Tuple[List[int], List[int]]:
    # Create masks for red and green colors using the specified thresholds
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    #color_threshold = 200  # Fine-tune this threshold depending on our images, higher mean less dummy points
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
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 9
    # Convert the image to the HSV color space for better color detection
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    # Define the lower and upper thresholds for red and green colors in HSV space
    lower_red = np.array([0, 100, 210])
    upper_red = np.array([10, 130, 240])
    lower_green = np.array([50, 100, 160])
    upper_green = np.array([100, 200, 255])
    # Find the red and green areas in the image
    red_x, red_y = find_ftl_single_color(c_image, hsv_image, kernel, lower_red, upper_red, 90)
    green_x, green_y = find_ftl_single_color(c_image, hsv_image, kernel, lower_green, upper_green, 200)
    # Filter points that are too close, because we want one point for each object
    min_distance = 50
    red_x, red_y = filter_close_points(red_x, red_y, min_distance)
    green_x, green_y = filter_close_points(green_x, green_y, min_distance)
    return red_x, red_y, green_x, green_y
# helper function for filtering too close points
def filter_close_points(x_coords: List[int], y_coords: List[int], min_distance: int) -> Tuple[List[int], List[int]]:
    filtered_x, filtered_y = [], []
    points = sorted(zip(x_coords, y_coords))
    if not points:
        return filtered_x, filtered_y
    prev_x, prev_y = points[0]
    filtered_x.append(prev_x)
    filtered_y.append(prev_y)
    for x, y in points[1:]:
        if np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2) >= min_distance:
            filtered_x.append(x)
            filtered_y.append(y)
            prev_x, prev_y = x, y
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
def test_find_tfl_lights(image_path: str, image_json_path: Optional[str]=None, fig_num=None):
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
def crop_traffic_lights(c_image, x_coordinates, y_coordinates, color: str, zoom_out_factor: int = 1):
    cropped_images = []
    window_height = int(zoom_out_factor * 80)
    window_width = int(window_height / 2.5)
    for x, y in zip(x_coordinates, y_coordinates):
        x_min, x_max = max(x - window_width // 2, 0), min(x + window_width // 2, c_image.shape[1])
        if color == 'red':
            y_min, y_max = max(y - 2 * window_height // 3, 0), min(y + 2 * window_height // 3, c_image.shape[0])
        else:  # green
            y_min, y_max = max(y - 2 * window_height // 3, 0), min(y + 2 * window_height // 3, c_image.shape[0])
        cropped_light = c_image[y_min:y_max, x_min:x_max]
        if cropped_light.size > 0:  # Check if the image is not empty
            cropped_images.append(cropped_light)
    return cropped_images
    return cropped_images
def display_cropped_images(cropped_images, title: str):
    num_images = len(cropped_images)
    if num_images == 0:
        print(f"No cropped images found for {title}.")
        return
    if num_images == 1:
        fig, ax = plt.subplots(figsize=(2, 2))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for ax, img in zip(axes, cropped_images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.suptitle(title)
    plt.show()
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
    # Load the new image
    # Load the new image
    new_image_path = '/Users/rashidab/Desktop/part_1/hamburg_000000_089696_leftImg8bit.png'
    new_image = Image.open(new_image_path)
    new_c_image = np.array(new_image)
    # Find the traffic lights in the new image
    new_red_x, new_red_y, new_green_x, new_green_y = find_tfl_lights(new_c_image)
    # Crop red and green traffic lights with the desired region
    zoom_out_factor_even_smaller = 2
    new_cropped_red_lights = crop_traffic_lights(new_c_image, new_red_x, new_red_y, color='red',
                                                                    zoom_out_factor=zoom_out_factor_even_smaller)
    new_cropped_green_lights = crop_traffic_lights(new_c_image, new_green_x, new_green_y,
                                                                      color='green',
                                                                      zoom_out_factor=zoom_out_factor_even_smaller)
    # Display the cropped red traffic lights
    display_cropped_images(new_cropped_red_lights, title="Cropped Red Traffic Lights")
    # Display the cropped green traffic lights
    display_cropped_images(new_cropped_green_lights, title="Cropped Green Traffic Lights")
if __name__ == '__main__':
    main()