# Human Activity Recognition Using Deep Learning

##  Overview
This project focuses on **time-series based Human Activity Recognition (HAR)** using wearable sensor data.  
We implemented and compared multiple deep learning architectures to evaluate their effectiveness in modeling spatiotemporal patterns:

- **Baseline Neural Network (NN)**  
- **Convolutional Neural Network (CNN)**  
- **Long Short-Term Memory (LSTM)**  

The goal was to benchmark these models for activity recognition tasks that are crucial in **healthcare monitoring, rehabilitation, and neuromuscular disorder analysis**.

---

## ⚙ Dataset
We used the **[UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)**, which contains **accelerometer** and **gyroscope** sensor readings from smartphones across **6 activity classes**:

- Walking  
- Walking Upstairs  
- Walking Downstairs  
- Sitting  
- Standing  
- Laying  

Each sample consists of **time-series motion signals**, segmented into windows that represent human activities.

---

##  Methodology

### 🔹 Data Preprocessing
- Normalized accelerometer & gyroscope values.  
- Segmented continuous signals into fixed-size windows.  
- Split dataset into **80% training** and **20% testing**.  

### 🔹 Models Implemented
1. **Baseline NN** – Fully connected layers for reference performance.  
2. **CNN** – Captures local spatiotemporal features from sensor signals.  
3. **LSTM** – Designed to capture long-term temporal dependencies in sequential data.  

---

##  Results

### Performance Table
| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Baseline NN  | 0.89     | 0.88      | 0.87   | 0.87     |
| CNN          | 0.94     | 0.93      | 0.94   | 0.94     |
| LSTM         | 0.82     | 0.80      | 0.81   | 0.80     |

---

## Analysis
- **CNN achieved the best performance** across all metrics (**Accuracy = 94%**).  
  - Effectively captured local signal variations.  
  - Provided robust feature extraction and generalization.  

- **LSTM underperformed (Accuracy = 82%)**  
  - Struggled with **overfitting** due to dataset size.  
  - Computationally expensive and less efficient for this dataset.  
  - Long-term dependencies were less relevant here.  

- **Baseline NN (Accuracy = 89%)**  
  - Decent performance but lacked ability to extract **complex temporal features**.  

---

## Key Takeaways
- **CNN is the most effective model** for this HAR dataset.  
- **LSTM may excel** in larger datasets with more meaningful temporal dependencies.  
- Demonstrates how **deep learning converts raw motion signals into actionable insights**, which is directly relevant to:  
  - **Healthcare AI**  
  - **Neuromuscular disorder monitoring**  
  - **Rehabilitation systems**  

---
