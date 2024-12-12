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

![Dataset Overview](PLACEHOLDER_FOR_IMAGE)

### Dataset Structure:
- **Source Satellites**: GFDM and CM-01.  
- **Data Diversity**: Images cover varied environments such as urban areas, oceans, fields, and mountainous terrains.  
- **Annotations**: Includes flight parameters such as altitude, speed, and direction.

### Unique Features:
1. **Multispectral Bands**: Different spectral responses enhance the ability to detect targets in diverse conditions.  
2. **Temporal Offset**: Exploits slight time differences across spectral bands for motion analysis.  
3. **High Resolution**: Enables precise localization and motion estimation.

---

## Methodology

![ASPIRE Framework](PLACEHOLDER_FOR_IMAGE)

### Components:
1. **Spatial Information Retention Branch (SIRB)**: Retains fine-grained spatial features crucial for small target detection.  
2. **Structured Mask Attention (SMA)**: Enhances detection accuracy by focusing on aircraft structural features.  
3. **Kalman Filter**: Estimates velocity, direction, and altitude based on temporal differences.

### Mathematical Basis:
The ASPIRE framework leverages advanced machine learning and statistical methods, including PCA for directional estimation and Kalman filtering for robust parameter retrieval.

---

## Experimental Results

### Detection Performance:
| Model         | mAP50 (%) | mAP (%) | Params (M) | GFLOPs |
|---------------|------------|---------|------------|--------|
| YOLOv8-s      | 99.0       | 78.4    | 10.8       | 27.3   |
| ASPIRE        | **99.3**   | **79.3**| 2.8        | 15.7   |

### Parameter Estimation:
- Velocity estimation error: <1%.  
- Altitude estimation bias: ~2,125 meters (35%).  

---

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
   



