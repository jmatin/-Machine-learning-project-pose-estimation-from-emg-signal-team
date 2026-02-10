```
├── competition
│   ├── Top 8 out of 42 groups.png
│   ├── team_submission.csv
│
├── doc
│   ├── paper.pdf
│   ├── Background.png
│   ├── Competition.png
│   ├── Dataset.png
│   ├── Overlapping windows.png
│   ├── Project guidelines 3-6.png
│   ├── Project guidelines6-7.png
│   ├── Simulation.png
│   └── joint angles.png
│
├── visualization
│   └── link
│
├── README.md
└── stat-ml-project.ipynb
 ```

<div align="center">
    <h1>Statistical Foundations of Machine Learning
        Project 2024 - 2025</h3>
</div>      

<div align="center">
    <h3>Muhammad Ehsan Siddique, Jordan Matin, Oussama Laakel</h3>
</div>

### Table of Contents

1. [Introduction](#Introduction)
2. [Dataset Overview](#Dataset-Overview)
3. [Task](#Task)
4. [Performance Metrics](#Performance-Metrics)
5. [Preprocessing](#Preprocessing)
   - [Signal Normalization](#Signal-Normalization)
   - [Feature Extraction](#Feature-Extraction)
   - [PCA](#pca)
6. [Baseline Approaches](#Baseline-Approaches)
   - [Support Vector Regressor (SVR)](#Guided-Gestures)
   - [Random Forest Regressor](#Free-Gestures)
   - [Discussion Section](#discussion-baseline)
7. [More Sophisticated Approach](#neural-network)
   - [CNN-LSTM](#cnn-lstm)
   - [MLP-Wavelet co-efficient](#mlp-wavelet)
   - [Discussion Section](#discussion-nn)
8. [Ensemble Strategies](#Ensemble-Strategies)
   - [Meta](#Averaging)
   - [Averaging](#Stacking) 
   - [Bias–Variance Trade-Off Analysis](#bias) 
   - [Discussion Section](#discussion-ensemble) 
9. [Results and Discussion](#Results-and-Discussion)
10. [Link to the Visualization](#visualization)

## [Introduction](#Introduction)
Understanding how our muscles move and control our hands is important in many fields, such as prosthetics, rehabilitation, and human-computer interaction. In this project, we aim to predict hand joint angles (also called degrees of freedom or DoFs) using electrical signals from muscles, known as surface Electromyography (sEMG).

The data is collected from sensors placed on the skin that measure muscle activity (sEMG), and a motion capture system records the actual movements of the hand joints. The goal is to build a machine learning model that can accurately predict the positions of 51 hand joints from the sEMG signals. This is a challenging task because the relationship between muscle signals and hand movement is complex and can vary across individuals and time.

## [Dataset Overview](#dataset-overview)

This project involves two datasets—Guided Gestures and Free Gestures—each containing multichannel surface electromyography (sEMG) signals recorded from a participant performing hand and finger movements. The goal is to predict 51 hand joint angles from the raw sEMG data.

🔹 Guided Gestures Dataset
This dataset includes structured and repetitive hand movements, ideal for model development and baseline evaluation.

- guided_dataset_X.npy: sEMG training data of shape (5, 8, 230000)
→ 5 sessions × 8 electrodes × 230,000 time samples

- guided_dataset_y.npy: Corresponding joint angle labels of shape (5, 51, 230000)
→ 5 sessions × 51 joint angles × 230,000 samples

- guided_testset_X.npy: Test data in shape (5, 332, 8, 500)
→ 5 sessions × 332 windows × 8 electrodes × 500 time samples
→ Predict 51 joint angles for each window (total: 5 × 332 = 1660 predictions)

🔹 Free Gestures Dataset
This dataset captures realistic, unstructured hand motions, making it more challenging and closer to real-world conditions.

- freemoves_dataset_X.npy: sEMG training data of shape (5, 8, 270000)

- freemoves_dataset_y.npy: Joint angle labels of shape (5, 51, 270000)

- freemoves_testset_X.npy: Test data of shape (5, 308, 8, 500)
→ Predict 51 joint angles for each of the 1540 (5 × 308) windows


##  [Task](#Task)

🔹 Signal Filtering (Optional)

🔹 Dataset preparation and augmentation through overlapping windows

🔹 Cross-validation strategy

🔹 Baseline approach

🔹 More sophisticated approach

🔹 Ensembling Strategies 

🔹 Final Prediction and Submission 

## [Performance Metrics](#Performance-Metrics)

To evaluate the quality of our predictions on a given test set, we use the following metrics:

---

### **Root Mean Squared Error (RMSE)**

As defined in the challenge description, RMSE is given by:

$$
\text{RMSE} = \sqrt{\frac{1}{N_{ts}} \sum_{i=1}^{N_{ts}} (y_i - \hat{y}_i)^2}
$$


Where:  
- $N_{ts}$ is the number of test observations  
- $y_i$ is the measured concentration of the $i^{\text{th}}$ test observation  
- $\hat{y}_i$ is the predicted concentration of the $i^{\text{th}}$ test observation
