# Deep Learning Course (CPSC 8430) Homework Projects

This repository contains all of my homework projects for the Deep Learning course (8430). The projects involve various deep learning concepts and techniques, primarily using the following technologies:

- **Python 3**
- **PyTorch**

## Homework Assignments

### HW1:

#### P1 - Deep vs Shallow:
1. Simulate a function.
2. Train on actual tasks using shallow and deep models.

#### P2 - Optimization:
1. Visualize the optimization process.
2. Observe the gradient norm during training.
3. Study what happens when the gradient is almost zero.

#### P3 - Generalization:
1. Network's Capacity to Fit Random Labels
2. Parameters vs Generalization
3. Exploring Flatness vs Generalization:
   - Part 1
   - Part 2

### HW2:Video caption generation

This project aims to create a system that automatically generates captions for short videos, describing their content. It uses a type of artificial intelligence called a sequence-to-sequence model to process the video and produce a series of words that form the caption.Here,result is evaluated with the BLEU score.



### HW3:Extractive Question Answering with Spoken-SQuAD

### Overview
In this project, I worked with a **Spoken-SQuAD** dataset, which is a spoken version of the original SQuAD dataset. The task is to build a model for extractive question answering.The code demonstrates a full workflow for question-answering using BERT, including model configuration, data preparation, training with mixed precision, and evaluation on both clean and noisy datasets. The results help assess the model's accuracy and resilience under varying noise conditions, highlighting areas where performance could potentially be improved.

### Dataset
- **Spoken-SQuAD Dataset**: Contains 37,111 question-answer pairs for training and 5,351 pairs for testing. The dataset includes different levels of white noise added to test the model's performance under challenging audio conditions.
- **Word Error Rate (WER)**: The training set has a WER of 22.77%, and the testing set has a WER of 22.73%.

### Tokenization
- The text is tokenized, meaning it is broken down into smaller units (tokens), which are then converted into IDs that the model can understand. 

### Performance Optimization
- **Simple**: Used sample code provided to start.
- **Medium**: Applied learning rate decay and experimented with the `doc_stride` parameter to adjust window size.
- **Strong**: Focused on improving preprocessing and try other pretrained models. Fine-tune postprocessing to handle edge cases.

### Model Training
- **Training Loop**: The model is trained over three epochs with the following steps:
- For each batch, the model calculates the loss and updates the weights through backpropagation. 
- The loss is tracked for each epoch, and a training accuracy placeholder is used.

### Evaluation Metrics & Results
   - **Exact Match (EM)**: Measures the percentage of predictions that exactly match the ground truth answer.
   - **F1 Score**: Considers precision and recall, measuring the overlap between the predicted and true answer text.



## HW4:Generative Model Training on CIFAR-10

### Overview
This project focuses on training three generative models—**DCGAN**, **WGAN**, and **ACGAN**—on the CIFAR-10 dataset. Each model is evaluated based on its ability to generate realistic and diverse images, utilizing unique techniques for improving generative performance.

### Dataset

- **CIFAR-10 Partial Subset:**
  - 32x32 color images resized to 64x64.
  - Classes include airplanes, cars, birds, cats, etc.
  - Used to minimize resource consumption.

### Tasks

1. **Train Generative Models:**
   - **DCGAN (Deep Convolutional GAN):** 
   - **WGAN (Wasserstein GAN):** 
   - **ACGAN (Auxiliary Classifier GAN):** 

2. **Evaluate Model Performance:**
   - Analyze training stability and image quality using metrics like Frechet Inception Distance (FID).
   - Compare generated images visually against real images.

3. **Visualize Training Progress:**
   - Track generator and discriminator losses.
   - Plot FID scores across epochs for each model.

### Outputs

1. **Generated Images:** 
   - Real vs. Fake image grids for visual comparison.
   - Models demonstrated varying levels of clarity and diversity.

2. **Evaluation Metrics:**
   - Generator and discriminator loss curves.
   - FID scores for all models over 50 epochs.






