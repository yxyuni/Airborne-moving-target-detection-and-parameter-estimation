# Multispectral Aerial Target Parameter Estimation

This repository presents **SHARP** (Spectral and High-dynamic Aerial Target with Flight Parameters), a novel benchmark dataset for multispectral aerial target detection and parameter estimation, alongside the **ASPIRE** (Aerial Spectral Parameter Intelligent Retrieval Estimator) framework. This project is aimed at addressing challenges in aerial target parameter estimation using advanced multispectral imaging and machine learning techniques.

---

## Highlights

- **SHARP Dataset**:  
  - Derived from the GFDM (8 bands, 450nm-900nm) and CM-01 (4 bands, 450nm-890nm) satellites.  
  - Includes **1,301 high-resolution images** with **512×512 pixel dimensions** and **2-meter ground resolution**.  
  - Provides detailed annotations, including velocity, direction, and altitude, for rigorous benchmarking.  

- **ASPIRE Framework**:  
  - Features a **Spatial Information Retention Branch (SIRB)** for small target detection.  
  - Integrates a **Structured Mask Attention (SMA)** mechanism for aircraft morphology detection.  
  - Leverages **Kalman filtering** for robust velocity estimation.

---

## Key Features

- **Multispectral Imaging**: Temporal differences across spectral bands enhance motion parameter estimation accuracy.
- **Comprehensive Benchmarking**: SHARP dataset sets a new standard for aerial target detection and parameter estimation.
- **State-of-the-Art Performance**: ASPIRE achieves significant improvements in detection accuracy and parameter estimation precision.

---

## Dataset Overview

![Dataset Overview](https://github.com/yxyuni/Airborne-moving-target-detection-and-parameter-estimation/blob/main/figs/77_final_combined.png)

### Dataset Structure:
- **Source Satellites**: GFDM-01 and CM-01.  
- **Data Diversity**: Images cover varied environments such as urban areas, oceans, fields, and mountainous terrains.  
- **Annotations**: Includes flight parameters such as altitude, speed, and direction.

### Unique Features:
1. **Multispectral Bands**: Different spectral responses enhance the ability to detect targets in diverse conditions.  
2. **Temporal Offset**: Exploits slight time differences across spectral bands for motion analysis.  
3. **High Resolution**: Enables precise localization and motion estimation.

---

## Methodology

![ASPIRE Framework](https://github.com/yxyuni/Airborne-moving-target-detection-and-parameter-estimation/blob/main/figs/fig-fullgraph-1.png)

### Components:
1. **Spatial Information Retention Branch (SIRB)**: Retains fine-grained spatial features crucial for small target detection.  
2. **Structured Mask Attention (SMA)**: Enhances detection accuracy by focusing on aircraft structural features.  
3. **Kalman Filter**: Estimates velocity, direction, and altitude based on temporal differences.

### Mathematical Basis:
The ASPIRE framework leverages advanced machine learning and statistical methods, including PCA for directional estimation and Kalman filtering for robust parameter retrieval.

---

# Experiments

## Detection Results

| **Model**           | **mAP₅₀ (%)** | **mAP₇₅ (%)** | **mAP (%)** | **Params (M)** | **GFLOPs** |
|----------------------|------------------|------------------|-----------|-------------|----------|
| Faster R-CNN        | 97.6            | 89.2            | 70.5      | 41.3        | 25.9     |
| RetinaNet           | 97.2            | 84.6            | 70.2      | 36.3        | 13.1     |
| CornerNet           | 97.8            | 85.7            | 69.8      | 201.0       | 254.0    |
| EfficientDet        | 97.0            | 83.7            | 69.0      | 3.8         | 2.3      |
| YOLOX               | 97.7            | 85.7            | 68.9      | 5.0         | 1.2      |
| TOOD                | 97.9            | 86.6            | 72.2      | 32.0        | 12.6     |
| RTMDet              | 97.9            | 88.5            | 71.5      | 4.78        | 1.3      |
| DINO                | 96.9            | 90.9            | 72.4      | 47.5        | 29.9     |
| DiffusionDet        | 96.9            | 87.7            | 69.3      | -           | -        |
| DDQ                 | 98.7            | 91.3            | 73.1      | -           | -        |
| YOLOv5-n            | 99.3            | 91.6            | 76.4      | 21.9        | 5.8      |
| YOLOv8-s            | 99.0            | 94.0            | 78.4      | 10.8        | 27.3     |
| YOLOv10-n           | 99.1            | 88.5            | 73.7      | 2.1         | 5.7      |
| YOLOv11-n           | 99.1            | 92.2            | 77.4      | 2.3         | 5.0      |
| YOLOv11-s           | 98.8            | 93.6            | 77.9      | 9.1         | 20.0     |
| **ASPIRE**          | **99.3**        | **96.3**        | **79.3**  | 2.8         | 15.7     |

## Experimental Setups

- **System Configuration**: Ubuntu 24.04 with NVIDIA GTX 4080 GPU using CUDA version 12.2.
- **Optimizer**: AdamW with a learning rate of 0.01, \(\beta = (0.9, 0.999)\), and a weight decay of 0.0005.
- **Training Settings**: 500 epochs on the SHARP dataset, split into 80% training and 20% validation sets.
- **Evaluation Metrics**: Accuracy, Recall, mAPₐ₅₀, and mAP across IoU thresholds from 0.5 to 0.95 (step size 0.05).

### Precision and Recall

Precision (P) and Recall (R) are defined as:

$$
P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}
$$

The area under the Precision-Recall curve represents the Average Precision (AP), with mAP calculated as:

$$
mAP = \frac{1}{N} \sum_{i=1}^{N} AP_{\text{IoU=0.5 + 0.05(i-1)}}
$$

## Target Detection Results

ASPIRE achieves the highest **mAPₐ₅₀** of 99.3%, outperforming YOLOv8-s (99.0%) and YOLOv11 (99.1%). Additionally, ASPIRE excels in stricter metrics such as **mAPₐ₇₅** (96.3%) and overall mAP (79.3%).

Despite its superior performance, ASPIRE maintains efficiency with only 2.8M parameters and 15.7 GFLOPs, balancing precision and computational requirements for real-world applications.

## Motion Parameter Estimation

Using a Kalman filter, ASPIRE estimates motion parameters such as apparent velocity, plane speed, and altitude with remarkable precision:

| **Parameter**             | **Average Error** | **Relative Error** |
|---------------------------|-------------------|--------------------|
| Apparent Velocity (x)     | 2.42 pixels       | 1%                 |
| Apparent Velocity (y)     | 2.49 pixels       | 1%                 |
| Plane Speed               | 178.58 km/h       | 7%                 |
| Flight Direction          | 7.55°            | 15%                |
| Altitude                  | 1,737m            | 25%                |

## Ablation Study

### Module Effectiveness

| **Model**    | **mAPₐ₅₀ (%)** | **mAPₐ₇₅ (%)** | **mAP (%)** | **Params (M)** | **GFLOPs** |
|--------------|------------------|------------------|-----------|-------------|----------|
| Baseline     | 98.8            | 90.8            | 76.6      | 2.7         | 6.3      |
| +SIRB        | 99.0            | 95.0            | 78.4      | 2.8         | 14.2     |
| +SMA         | 98.9            | 92.2            | 77.3      | 2.8         | 8.3      |
| **ASPIRE**   | **99.3**        | **96.3**        | **79.3**  | 2.8         | 15.7     |

### Backbone Study

| **Backbone (-n)** | **mAPₐ₅₀ (%)** | **mAPₐ₇₅ (%)** | **mAP (%)** | **Params (M)** | **GFLOPs** |
|-------------------|------------------|------------------|-----------|-------------|----------|
| YOLOv5            | 99.3            | 96.0            | 79.0      | 2.3         | 14.5     |
| YOLOv8            | 99.3            | 96.3            | 79.3      | 2.8         | 15.7     |
| YOLOv10           | 99.3            | 92.8            | 75.4      | 2.1         | 13.1     |
| YOLOv11           | 99.1            | 94.4            | 78.2      | 2.3         | 16.9     |

ASPIRE demonstrates consistent improvements across backbones, achieving a balance between accuracy and computational efficiency.


## Installation and Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yxyuni/Airborne-moving-target-detection-and-parameter-estimation.git
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:

   ```bash
   python demo.py --input <input_image>
   ```

4. **Train the model**:

   ```bash
   python train.py --config config.yaml
   ```

5. **Evaluate the model**:

   ```bash
   python evaluate.py --model checkpoints/model.pth --data <dataset_path>
   ```

6. **Visualize results**:

   ```bash
   python visualize.py --model checkpoints/model.pth --input <input_image>
   



