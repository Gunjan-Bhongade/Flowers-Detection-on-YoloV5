
# Flowers Detection on Yolov5

## Aim And Objectives

### Aim

The aim of this project is to develop a robust and efficient flower detection system using the YOLOv5 (You Only Look Once version 5) object detection algorithm. The system should accurately identify and classify different types of flowers in images and real-time video streams, providing a valuable tool for applications in horticulture, agriculture, environmental monitoring, and educational purposes.

### Objectives

1. Literature Review:

- Conduct a comprehensive review of existing methods for flower detection and classification.
- Study the YOLOv5 algorithm, its architecture, and its advantages over other object detection methods.

2. Dataset Collection and Preparation:

- Collect a diverse dataset of flower images representing different species and variations.
- Annotate the dataset with bounding boxes and labels for each flower.
- Perform data augmentation to increase the diversity and size of the dataset.

3. Model Training:

- Configure the YOLOv5 model for flower detection by adjusting its hyperparameters and architecture.
- Train the YOLOv5 model using the prepared dataset, ensuring adequate validation and testing splits.
- Monitor training performance using metrics such as precision, recall, mean Average Precision (mAP), and loss.

4. Model Evaluation:

- Evaluate the trained YOLOv5 model on a separate test set to assess its performance.
- Analyze the model's accuracy, detection speed, and robustness to different conditions (e.g., lighting, occlusion, and background clutter).

5. Optimization and Fine-tuning:

- Fine-tune the model by adjusting hyperparameters, adding more training data, or applying transfer learning if necessary.
- Implement techniques to optimize the model for real-time detection, such as model pruning and quantization.

### Abstract

This project explores the application of the YOLOv5 (You Only Look Once version 5) algorithm for the detection and classification of various flower species in images and video streams. Leveraging the speed and accuracy of YOLOv5, we aim to develop a robust system capable of identifying different types of flowers in diverse conditions. This system has potential applications in horticulture, agriculture, environmental monitoring, and education. Through the collection and annotation of a comprehensive flower dataset, rigorous model training and evaluation, and the deployment of a real-time detection application, we demonstrate the efficacy of YOLOv5 in handling the complexities of flower detection. Our results indicate that YOLOv5 provides a promising solution for real-time flower identification, contributing to advancements in computer vision and AI-driven plant recognition systems.

### Introduction

Flower detection is a significant task in various domains, including horticulture, agriculture, environmental monitoring, and education. Accurate identification and classification of flowers can aid in plant health monitoring, biodiversity studies, and the automation of agricultural processes. Traditional methods for flower detection often involve manual identification, which can be time-consuming and prone to errors. Recent advancements in computer vision and deep learning have paved the way for automated and efficient flower detection systems.

YOLOv5 (You Only Look Once version 5) is one of the most advanced and efficient object detection algorithms available today. Known for its speed and accuracy, YOLOv5 can detect objects in real-time, making it an ideal choice for applications requiring fast and reliable detection. Unlike traditional object detection methods, YOLOv5 processes the entire image with a single neural network, resulting in a significant reduction in computation time and an increase in detection accuracy.

In this project, we aim to harness the capabilities of YOLOv5 to develop a flower detection system that can accurately identify and classify different types of flowers in various conditions. The objectives of this project include collecting and annotating a diverse dataset of flower images, training the YOLOv5 model, evaluating its performance, and deploying a real-time flower detection application. By achieving these objectives, we aim to demonstrate the potential of YOLOv5 in enhancing flower detection and contributing to advancements in computer vision and AI-driven plant recognition systems.

The following sections of this report will detail the methodology employed in collecting and annotating the flower dataset, the training and evaluation of the YOLOv5 model, the implementation of the real-time detection application, and the results obtained from the experiments. Through this project, we seek to provide a comprehensive solution for flower detection that can be utilized in various practical applications, ultimately contributing to the field of computer vision and artificial intelligence.

### Literature Review:

The field of flower detection and classification has seen significant advancements with the advent of deep learning and convolutional neural networks (CNNs). Traditional methods relied heavily on handcrafted features and classical machine learning algorithms, which were often limited in their ability to handle complex and diverse datasets.

### Jetson Nano Compatibility:

NVIDIA Jetson Nano is a compact and powerful platform designed for AI edge computing. It supports a range of AI frameworks, including TensorFlow, PyTorch, and YOLO, making it suitable for deploying deep learning models like YOLOv5.

1. Hardware Specifications:

- CPU: Quad-core ARM Cortex-A57 MPCore processor
- GPU: 128-core Maxwell GPU
- Memory: 4GB LPDDR4
- Storage: microSD slot for main storage

2. Software Environment:

- Operating System: Ubuntu-based JetPack SDK, which includes libraries and APIs for deep learning, computer vision, and multimedia processing.
- YOLOv5 Compatibility: YOLOv5 can be optimized to run on Jetson Nano by leveraging NVIDIA’s TensorRT for accelerated inference. This involves converting the trained YOLOv5 model into a TensorRT engine, significantly enhancing the inference speed on the Jetson Nano.

### Proposed System:

The proposed flower detection system using YOLOv5 on Jetson Nano involves several key components:

1. Data Collection and Annotation:

- Dataset Creation: Collect a diverse dataset of flower images representing various species. Ensure the dataset includes variations in lighting, background, and occlusion.
- Annotation: Annotate the dataset with bounding boxes and labels for each flower using tools like LabelImg or Roboflow.

2. Model Training:

- Configuration: Configure YOLOv5 with appropriate hyperparameters for flower detection. This includes defining the network architecture, learning rate, batch size, and epochs.
- Training: Train the YOLOv5 model using the annotated dataset, leveraging techniques like data augmentation and transfer learning to improve performance.

### Methodology

 Traditional Methods Early object detection techniques relied heavily on hand-crafted features and traditional machine learning algorithms. Notable methods include:

- Histogram of Oriented Gradients (HOG): Introduced by Dalal and Triggs (2005), HOG is a feature descriptor used to detect objects, primarily humans. It works by calculating gradient orientations in a dense grid of uniformly spaced cells and using overlapping local contrast normalization.

- Deformable Part Models (DPM): Proposed by Felzenszwalb et al. (2008), DPM uses a set of parts and a deformable spatial model to capture the variability in object shapes. It employs a sliding window approach for detection, which is computationally expensive. vim ~/.bashrc

### Installation
#### Initial Configuration

sudo apt-get remove --purge libreoffice* sudo apt-get remove --purge thunderbird*

#### Create Swap
udo fallocate -l 10.0G /swapfile1 sudo chmod 600 /swapfile1 sudo mkswap /swapfile1 sudo vim /etc/fstab

#### make entry in fstab file
/swapfile1 swap swap defaults 0 0

#### Cuda env in bashrc
vim ~/.bashrc

#### add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

#### Update & Upgrade
sudo apt-get update

sudo apt-get upgrade

#### Install some required Packages
sudo apt install curl curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py sudo python3 get-pip.py sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow

#### Install Torch
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" sudo python3 -c "import torch; print(torch.cuda.is_available())"

#### Install Torchvision
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision cd torchvision/ sudo python3 setup.py install

#### Clone Yolov5
git clone https://github.com/ultralytics/yolov5.git cd yolov5/ sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1 sudo pip3 install -r requirements.txt

#### Download weights and Test Yolov5 Installation on USB webcam
sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt --source 0

### Advantages:
#### 1. Real-Time Detection:

- YOLOv5 is designed for real-time object detection, making it highly suitable for applications requiring immediate results, such as live video analysis.
#### 2. High Accuracy:

- YOLOv5’s architecture and optimization techniques enable high detection accuracy, even with small objects like flowers in diverse and complex backgrounds.
#### 3. Efficiency:

- The model processes the entire image in a single forward pass, significantly reducing computation time compared to traditional methods.
#### 4. Scalability:

- YOLOv5 can be easily scaled and adapted for different hardware platforms, including powerful GPUs and edge devices like the Jetson Nano.
#### 5. Versatility:

- The system can be applied to a wide range of flower species and can be extended to other object detection tasks with minimal modifications.

### Applications:
- Horticulture and Agriculture:

Automated flower detection can assist in monitoring plant health, identifying pests and diseases, and optimizing crop management practices.

- Environmental Monitoring:

Detecting and classifying flowers in natural habitats can aid in biodiversity studies and conservation efforts.

- Educational Tools:

The system can be used in educational applications to help students and researchers identify and learn about different flower species.

- Floral Industry:

Automated sorting and grading of flowers based on species and quality can improve efficiency in the floral supply chain.

- Smart Gardens:

Integration with smart garden systems for real-time monitoring and maintenance of flower beds and ornamental plants.

### Future Scope:
- Model Enhancement:

Further improve the model’s accuracy and robustness by incorporating advanced techniques like self-supervised learning and few-shot learning.

- Edge Deployment:

Optimize the system for deployment on various edge devices beyond Jetson Nano, such as smartphones and IoT devices.

- Multispectral Imaging:

Integrate multispectral or hyperspectral imaging to enhance flower detection and classification accuracy under different environmental conditions.

- Expanded Dataset:

Continuously expand and diversify the dataset to include more flower species and variations to improve the model’s generalizability.

- User-Friendly Applications:

Develop more user-friendly interfaces and applications for end-users, including mobile apps and web-based platforms.

### Conclusion:
The development of a flower detection system using YOLOv5 demonstrates the potential of advanced deep learning algorithms in achieving high accuracy and efficiency in real-time object detection tasks. By leveraging the capabilities of YOLOv5 and optimizing it for edge devices like the Jetson Nano, this project provides a practical solution for various applications in horticulture, agriculture, environmental monitoring, and education. Future advancements and continuous improvements in the model and deployment strategies will further enhance the system's capabilities and broaden its applicability.

### References:
1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/search/2/image?phrase=helmet

3] Google images

### Articles:
1] https://www.bajajallianz.com/blog/motor-insurance-articles/what-is-the-importance-of-wearing-a-helmet-while-riding-your-two-wheeler.html#:~:text=Helmet%20is%20effective%20in%20reducing,are%20not%20wearing%20a%20helmet.

2] https://www.findlaw.com/injury/car-accidents/helmet-laws-and-motorcycle-accident-cases.html










