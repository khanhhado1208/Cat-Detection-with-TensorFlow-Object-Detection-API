# Cat Detection with TensorFlow Object Detection API

This project demonstrates how to build an object detection model to detect **cats** using the **TensorFlow 2 Object Detection API**. 

The model is trained on a custom dataset of 200 labeled cat images, fine-tuned from a **Faster R-CNN ResNet101 640x640** base model pretrained on the COCO dataset.

## Overview

- Objective: Detect cats in images using bounding boxes
- Model: Faster R-CNN with ResNet101 backbone
- Dataset: 200 cat images (JPG) labeled in Pascal VOC format
- Tools: TensorFlow 2, Roboflow, TensorBoard, CUDA (GPU)
- Result: Accurate detection with clean bounding boxes (100% accuracy on test set)

## Dataset
- 200 labeled cat images from Google
- Bounding boxes labeled using **Roboflow** 
- Converted from `.xml` to `.record` (TFRecord format)
- Splits:
  - Train: 150 images
  - Validation: 20 images
  - Test: 30 images
 
  ## Tools & Setup

- TensorFlow 2.x + Object Detection API
- Anaconda Virtual Environment
- TensorBoard for monitoring loss
- GPU acceleration (CUDA 11.2, cuDNN 8.1.0)

  ## Model Training Pipeline

1. Download pre-trained **Faster R-CNN ResNet101 640x640** model from TensorFlow Model Zoo
2. Modify `pipeline.config` to match custom dataset & TFRecord paths
3. Train model for ~25,000 steps
4. Monitor loss metrics via **TensorBoard**
5. Save exported model and evaluate on test set

  
## Training Insights

- Loss functions (classification, localization, objectness) all decreased smoothly
- Learning rate decay confirmed stable optimization
- Training tracked using **TensorBoard**
  
![image](https://github.com/user-attachments/assets/1aa5b21e-b8c1-44eb-aa00-79d0f6ad45e8)

![image](https://github.com/user-attachments/assets/f29e642f-5622-485c-9c72-3cb9af49caf0)


## Inference Example

Run inference on 5 random test images. The model draws bounding boxes around detected cats with high confidence.

![image](https://github.com/user-attachments/assets/36fca0c8-13c8-428f-8f8b-485214d0ccc1)

![image](https://github.com/user-attachments/assets/311ea9d1-2290-48ca-b17a-61f98c3c549b)

![image](https://github.com/user-attachments/assets/c358bd61-f1b8-4804-a670-60b691962db3)

![image](https://github.com/user-attachments/assets/e3d3dd84-abd0-4c44-ba97-17b1a8c70561)

![image](https://github.com/user-attachments/assets/14a32527-215f-4986-9d33-2537fa7c79d2)

## Results

- 100% confidence in detection on test set
- Bounding boxes clearly show cat positions
- Smooth loss curves = good convergence
- Integrated inference visualization in notebook

## Future Improvements

- Add more data for better generalization
- Try alternative models (e.g. RetinaNet, EfficientDet)
- Apply image augmentation to increase diversity
