# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# # Load the image using PIL
# image_path = 'images_set/aachen/aachen_000092_000019_leftImg8bit.png'
# image = Image.open(image_path)
#
# # Example coordinates for the rectangles
# x_coords = [100, 200, 300]
# y_coords = [50, 150, 250]
# width, height = 50, 50  # Width and height of the crop
#
# # Create a new figure to plot the crops
# fig, axes = plt.subplots(1, len(x_coords), figsize=(15, 5))
#
# for i, (x, y) in enumerate(zip(x_coords, y_coords)):
#     # Define the coordinates for the rectangle to crop
#     left = x
#     top = y
#     right = x + width
#     bottom = y + height
#
#     # Crop the image
#     cropped_image = image.crop((left, top, right, bottom))
#
#     # Display the cropped image in the plot
#     axes[i].imshow(cropped_image)
#     axes[i].axis('off')
#
#     # Add a hyperlink to the image if it's hosted online
#     # Replace 'url_to_image' with the actual URL
#     axes[i].set_title("Link to Image", color='blue')
#     axes[i].title.set_url('url_to_image')
#
# plt.show()
