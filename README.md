# Traffic Light Detection Project

This project focuses on traffic light detection using image processing techniques and neural networks. 

## Workflow
The workflow begins with a high-pass filter convolution for image enhancement and a conversion to the HSV color space. After this step, red and green filters are applied to isolate potential traffic lights, resulting in crops of the suspected traffic lights. These cropped images are then provided as input to the neural network for classification. The primary purpose of the neural network is to determine if these isolated regions indeed represent traffic lights, making it a pivotal component of the traffic light detection process.

## Additional Information
For additional images, detailed explanations of our code, and to see the results in action, please refer to the attached presentation named [final_presentation.pptx](./final_presentation.pptx).
