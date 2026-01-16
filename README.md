<a id="top"></a>

# Car Damage Classification, Detection and Captioning
*An end-to-end **computer vision system** that **detects** whether a car is damaged, **localizes** the damaged area, **estimates** damage severity, and generates a **natural-language description** from a single input image.*

<details>
<summary><strong>Table of Contents</strong></summary>

- [Demo](#demo)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Tech Stack (Requirements)](#tech-stack-requirements)
- [Initialization](#initialization)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Instructions](#dataset-instructions)
- [Project Structure](#project-structure)
- [Process](#process)
- [Results](#results)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)
- [Contact](#contact)
- [License](#license)

</details>

---

# Demo

<p align="center">
  <img src="assets/demo.gif" alt="Streamlit Demo" width="700"><br>
  <em>Streamlit demo showcasing end-to-end car damage computer vision system</em>
</p>

Steps to use this demo:<br>
<ol>
    <li>Tick the box at the top of the page if the user has a CUDA-enabled GPU</li>
    <li>Upload the car image by clicking the "browse files" button</li>
    <li>Click on the "run inference" button and it will output the intended prediction.
    </li>
</ol>

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Key Features

- **Damage Presence Classification**<br>Custom ResNet50 Classifier to determine whether the vehicle is damaged or not
- **Damage Location Detection**<br>Yolov8 to localize damage regions on the vehicle and output the bounding box and class label for the damaged areas.
- **Damage Severity Estimation**<br>Yolov8 to estimate the damage severity on the vehicle and output the bounding box where the damage severity is classified from
- **Image Captioning**<br>Vision Transformer + GPT2 to generate descriptive captions about the car.
- **Multi-Stage Inference Model**<br>Compile all models into one inference model to produce an intelligent inference flow
- **Streamlit Web Application**<br>A python-based library for real-time inference


<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Model Architecture
1. Car Image --> Custom Resnet50 classifier
2. If damaged --> Yolov8 damage localizer + Yolov8 damage severity estimator
3. Caption generation using ViT + GPT2

```text
                                      ├─► True ─► Yolov8 models ┐
Image --> Custom Resnet50 Classifier ─|                         ├─► Caption
                                      ├─► False ────────────────┘
```
<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Tech Stack (Requirements)
```markdown
- torch=2.9.1
- streamlit=1.52.2
- torchvision=0.24.1
- pillow=12.0.0
- transformers=4.57.3
- skorch=1.3.1
- pandas=2.3.3
- numpy=2.3.5
- seaborn=0.13.2
- matplotlib=3.10.7
- tqdm=4.67.1
- scikit-learn=1.8.0
- scipy=1.16.0
- Ultralytics=8.4.0
- python=3.11.14
```
```text
Notes: 
1. All requirements are provided in the requirements.txt file
2. Should users have an NVIDIA GPU that supports cuda, replace the following tech stack requirements
- torch=2.9.1+cu128
- torchvision=0.24.1+cu128
```
<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Initialization

The trained model weights are **not included in this repository** due to GitHub file size limitations.

## Download Instructions

1. Download the model weights from Google Drive:  
   👉 **[Download CAR DAMAGE MODEL.zip](https://drive.google.com/file/d/14fE8VRGD1JF7qdHt9BqXlsBdoJssnBuA/view?usp=drive_link)**

2. After downloading, unzip the folder place the file in the following directory:
```text
models/
└── inference_model/
```

## Dataset Instructions

This project uses a retinal fundus dataset with clinical captions.

Due to licensing and privacy restrictions, the dataset cannot be redistributed.

### How to obtain the dataset
1. Download from: <br>[\[Dataset Source 1\]](https://www.kaggle.com/datasets/anujms/car-damage-detection) <br> [\[Dataset Source 2\]](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset) <br>[\[Dataset Source 3\]](https://drive.google.com/file/d/1uqFSmoea53oi163XhiPNjZJ7Y_3U35WZ/view?usp=sharing)
2. Place files in: ```data/original_data```

### Setup
1. In the ```src/tools/``` folder, run the ```dataset_curation.py``` file. Ensure that the dataset source is installed properly in the correct directory!

2. After that, run the ```split_damage_undamaged.py```, ```dataset_splitting.py``` file and ```YOLO_label_splitting.py``` file to split into train, test, and validation as well as format the data to follow YOLO standards.

3. Run the ```csv_splitting.py``` to get the individual pieces of the whole data.

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div> 

---

# Installation
1. Clone the github repository using the command below in the command prompt/ terminal
```bash
git clone https://github.com/ReubenZSTechs/Car-Damage-Classification-Detection-And-Captioning.git
```

2. Locate the folder
```bash
cd Car-Damage-Classification-Detection-And-Captioning
```

3. Install the virtual environment using the requirements.txt file
```bash
pip install -r requirements.txt
```

4. Should advance users wish to use CUDA-supported NVIDIA GPU, users can delete the pytorch and torchvision installation from the ```requirements.txt``` and run the command below in the command prompt/ terminal

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Usage
1. Verify whether the streamlit library is installed properly
```bash
streamlit --version
```

2. If streamlit library is installed, run the following command in the command prompt
```bash
streamlit run app.py
```

3. The command will bring you to the website using your default browser.

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Project Structure
```text
├── app.py
├── inference.py
├── models/
│   └── final_model/
│       └── caption_model/
│       └── classification_model/
│           └── resnet50_classifier.pth
│       └── damage_location_model/
│           └── yolov8_damage_location_best.pt
│       └── damage_severity_model/
│           └── yolov8_damage_severity_best.pt
│   └── inference_model/
│       └── caption_model/
│       └── meta.pt
│       └── resnet_damage.pth
│       └── yolo_location.pt
│       └── yolo_severity.pt
│   └── runs/
│       └── captioning_model/
│           └── ViT-GPT2
│       └── damage_models/
│           └── yolov8_damage_location_model/
│           └── yolov8_damage_lseverity_model/
├── assets/
│   └── demo.gif
├── data/
│   └── car_classification_resnet_data/
│       └── testing/
│       └── training/
│       └── validation/
│       └── labels.csv
│   └── dataset_csv_file/
│       └── dataset_captionining.csv
│       └── dataset_filepath.csv
│       └── dataset_labeling.csv
│       └── dataset.csv
│   └── dataset_yaml_file/
│       └── images/
│       └── labels/
│       └── data_location.yaml
│       └── data_severity.yaml
│   └── images/
│   └── original_data/
│       └── archive (1)/
│           └── img/
│           └── test/
│           └── train/
│           └── val/
│       └── data1a/
│           └── training/
│           └── validation/
│       └── Roboflow_annotation/
│           └── data/
│               └── images/
│               └── labels/
│           └── data.yaml
│           └── README.dataset.txt
│           └── README.roboflow.txt
│   └── YOLO_DATASET_LOCATION/
│       └── images/
│       └── labels/
│       └── data_location.yaml/
│   └── YOLO_DATASET_SEVERITY/
│       └── images/
│       └── labels/
│       └── data_severity.yaml/
├── src/
│   └── model_training/
│       └── CAPTIONING_MODEL.ipynb
│       └── FINAL_MODEL_BUILDING.ipynb
│       └── LOCALIZATION_MODEL - LOCATION.ipynb
│       └── LOCALIZATION_MODEL - SEVERITY.ipynb
│       └── RESNET50_MODEL.ipynb
│   └── tools/
│       └── csv_splitting.py
│       └── dataset_curation.py
│       └── dataset_splitting.py
│       └── labeling_dataset.py
│       └── split_damage_undamaged.py
│       └── YOLO_label_splitting.py
├── requirements.txt
└── README.md
```

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Process
1. Initialize the model by running the python files first to generate the csv and splitting the images into training, testing, and validation.
2. Train and test the custom resnet50 model to detect whether the car is damaged or not.
3. Fine-tune two YOLO models to localize the damage and estimate the severity of the damage.
4. Fine-tune vision transformer + GPT2 model for captioning.
5. Collate all model results in a single model and simplify for inferencing.

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Results
**Custom Resnet50 Model**<br>
Accuracy: 91.3%<br>
ROC-AUC (Undamaged VS damaged): 97.05%<br>
PR-AUC (Undamaged VS damaged): 97.49%<br>

**YOLO Localization Model**<br>
front: mAP50-95 = 0.1523<br>
back: mAP50-95 = 0.0912<br>
rear-left: mAP50-95 = 0.0481<br>
rear-right: mAP50-95 = 0.0446<br>

**YOLO Severity Model**<br>
low: mAP50-95 = 0.0304<br>
medium: mAP50-95 = 0.0305<br>
high: mAP50-95 = 0.1959<br>

**Captioning Model**<br>
Test Loss: 1.0752<br>

```text
Notes: All results and the training process of the model is available on model_training folder (See Project Structure for location)
```

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Future Work
1. Using a better model than YOLOv8 to improve the localization and the severity estimation.
2. Fine-tune a BLIP captioning model to provide a better captioning feature
3. Use a faster loading library to speedup initialization
4. Use docker for mobile inference
5. Gather more data to improve model's performance for classification, damage localization and damage severity estimation

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Disclaimer
This project is for **research and educational purposes only**.

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Contact
Name    : Reuben Zachary Susanto<br>
Contact : reuben.zachary.rz@gmail.com

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# License
Distributed under the MIT License. See ```LICENSE``` for more information

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>
