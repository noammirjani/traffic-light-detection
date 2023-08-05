# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import subprocess
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
#
# def on_click(event):
#     for i, ax in enumerate(axes):
#         if event.inaxes == ax:
#             subprocess.run(['open', image_path]) # For Mac
#             # subprocess.run(['xdg-open', image_path]) # For Linux
#             # subprocess.run(['start', image_path], shell=True) # For Windows
#             break
#
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
#     # Title
#     axes[i].set_title("Click to View Full Image", color='blue')
#
# # Connect the click event to the on_click function
# fig.canvas.mpl_connect('button_press_event', on_click)
#
# plt.show()
