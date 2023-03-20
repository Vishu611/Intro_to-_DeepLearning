# Final Assignment 

# Monocular Mask and Depth Estimation Model
  
Using CNN to predict the object masks and depths from a single image


Team members:

- Deeksha Pandit- dp35222
- Kai Zhang- kz3967
- Meghna Pudupakkam Mukesh- mp49543
- Shivangi Dubey- scd2422
- ViswaTej Seela- VS26276

## Problem and Motivation

Camera surveillance is an essential application of computer vision in which the goal is to monitor and detect activities in a given scene. A common challenge in camera surveillance is to separate the foreground (i.e., the moving objects) from the background (i.e., the stationary objects) and estimate the depth of the objects in the scene.

Traditional approaches to this problem involve hand-crafted feature extraction and model-based methods, which are often labor-intensive and time-consuming. However, recent advancements in deep learning, specifically convolutional neural networks (CNNs), have shown promising results in various computer vision tasks, including object detection and semantic segmentation.

To address the challenges of camera surveillance, we propose building a CNN model that takes the background and foreground images as inputs and predicts the depth and mask of the objects in the scene. The background image provides information about the stationary objects in the scene, while the foreground image provides information about moving things.

The proposed CNN model will enable efficient and accurate separation of foreground and background objects in camera surveillance, essential for detecting and tracking objects in real-time. Additionally, it will allow for precise depth estimation, which can be used for a wide range of applications, such as object detection and tracking, activity recognition, and scene reconstruction.

Data Source
Our model is supposed to take two images, background (bg) and fg_bg, as inputs at once. The output should be the mask, which is the segmented humans in the fg_bg, and the depth map of the fg_bg. The images are of dimension 224x224, both for inputs and outputs.


## Augmentations:

We started by using the resize function on the 64x64 sized images. Then we changed the function to resize the images to 224x224, as we wanted to experiment with smaller resolution images before moving on to larger ones (transfer learning). Additionally, we added the ColorJitter() function for data augmentation. We normalized the images using the mean and standard deviation of Fg-Bg images that we had calculated and then converted the images to tensors using the ToTensor() function.

## Model
First, ResNet18 architecture was used and the average pooling and FC layer were removed. Padding was set to 1 to maintain the same input and output size, but this led to a memory issue due to the large amount of memory needed for processing. It was then realized that not only parameter count but also memory storage and forward/backward pass memory were important considerations.

So after trying various models available, we chose to work with the UNet model.

UNet is a popular CNN model for image segmentation tasks that separates foreground objects from the background. It uses an encoder-decoder architecture with skip connections to generate high-resolution segmentation masks from input images. The UNet model can be used for tasks such as object detection and tracking, activity recognition, and scene reconstruction.

There is no dense layer in the UNet model, so images of different sizes can be used as input — So we first tried running the model on 64x64 size images and then switched to the 224x224 images of our dataset (transfer learning)


For our assignment, we have made a few changes to the UNet model:

First, we have some common layers for both depth and mask images. Then we split the layers into two — one set for mask and another set for depth images. While running the model for depth images, we first froze the mask layers which were newly created then ran the model and obtained the results. While running the model for mask images, we froze the depth layers which were newly created and ran the model to obtain results

Finally, the model would return logits_mask if we are running the model for mask images or it would return logits_depth if we are running the model for depth images.

## Loss Functions:

MSELoss() — Did not give good results

BCEWithLogitsLoss() — This worked best for mask images and this is the loss function we used for mask images, but for depth images, it gave this kind of output even after a few epochs: BCEWithLogitsLoss_ForDepth

L1Loss() — Then we realized we had to use different loss functions for both depth and mask images and planned to go with L1Loss() for depth images and this worked well

## Results

Intersection over Union (IOU)

This metric has been used to evaluate masks. The Intersection-Over-Union (IoU), also known as the Jaccard Index, is one of the most commonly used metrics in semantic segmentation, and for good reason. The IoU is a very straightforward metric that’s highly effective.

IoU is the area of overlap between the predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth as shown in the image. This metric ranges from 0–1 (0–100%) with 0 signifying no overlap and 1 signifying perfectly overlapping segmentation.

Root Mean Squared Error (RMSE)

Root Mean Square Error (RMSE) is a commonly used metric for evaluating the performance of depth prediction models. RMSE provides a quantitative measure of how well a depth prediction model is performing in terms of the difference between the predicted and ground truth depth values. Lower RMSE values indicate better performance, as the predicted values are closer to the ground truth values.

RMSE can be used to compare the performance of different depth prediction models or to track the performance of a single model over time as it is trained and tested on different datasets.

IoU Results: Average Metric = 0.8109732806682587
RMSE Results: Average Metric = 0.6715668320655823

## Conclusion

However, it is important to note that monocular mask and depth estimation models have some limitations, such as their inability to handle transparent or reflective objects, and their potential failure in highly textured or repetitive scenes. Additionally, these models may not generalize well to new scenes or objects, as they are trained on a specific dataset.

In conclusion, our monocular mask and depth estimation model using semantic segmentation is a promising approach for various computer vision applications, and the obtained results demonstrate its effectiveness in predicting segmentation and depth maps for input images of 224x224 dimensions. Further research can be conducted to overcome the limitations of these models and improve their generalization capabilities.

Thank you for reading :)
