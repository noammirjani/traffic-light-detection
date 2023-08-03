# mobileye-team_2
mobileye-team_2 created by GitHub Classroom
Team:
- Yehuda Heller
- Noam Mirjani
- AboUnis Rashid


## Project Description
The project is about detecting traffic lights (red and green) in images taken from a car camera.

## Project Structure
the steps of the search are:
for each image:
* find lights:
1. Preprocessing: The input image is subjected to maximum filtering to enhance important features and edges. Gaussian blur is applied to reduce noise and smooth the image. A specific kernel is defined for image processing operations. 
2. Color Space Conversion: The image is converted to the HSV color space. This conversion simplifies color detection tasks for specific colors.
3. Color Detection: It isolates pixels within the specified lower and upper color range in the HSV color space and performs morphological closing to smooth and fill gaps in the detected regions. Finally, contours of objects are detected in the combined mask, and the x and y coordinates of the detected color points are computed by taking the mean of the contour points along the corresponding axes.
4. Point Filtering: Detected points that are too close to each other are filtered out to avoid multiple detections of the same traffic light. A minimum distance threshold is used to ensure only one point is kept per traffic light. 

* crop images: 

for now the results of finding the traffic lights are good but not perfect, we are working on improving the results.
All the green and red lights are detected, but there are some false positives that means that we detect too red and green 
lights that are not a traffic light. as far as we tested for now we do not have false negative that means that we do not miss traffic lights.