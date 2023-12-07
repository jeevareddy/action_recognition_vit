# action_recognition_vit

Action Recognition using TimeSformer Transformer Architecture
This repository contains code and resources for exploring the efficacy of the TimeSformer transformer architecture for action recognition tasks using the HMDB (Human Motion Database) dataset. The study investigates two variants of the TimeSformer model and explores their performance in action recognition.

Overview
The primary goal of this project is to assess the effectiveness of transformer-based models, specifically the TimeSformer architecture, for action recognition in videos. The repository provides code for pre-processing videos into spatiotemporal sequences, extracting frame-level features, and evaluating the performance of both base and high-resolution (HR) TimeSformer models.

Contents
preprocessing/: Code for preprocessing video data and extracting frame-level features
TimeSformer_base_model/: Implementation of the base TimeSformer model
TimeSformer_HR_model/: Implementation of the high-resolution TimeSformer model
experiments/: Scripts and notebooks for conducting experiments and evaluating model performance
results/: Storage for experiment results, performance metrics, and analysis

Data Sources
The project utilizes the HMDB dataset, which consists of a diverse collection of videos encompassing various human actions across different categories. Each video in the dataset is labeled with the corresponding action category, providing ground truth annotations for evaluation purposes.

Methodology
Model Variants: Compare the base TimeSformer model against the high-resolution (HR) variant to understand the impact of incorporating additional spatial information on action recognition performance.
Hyperparameter Tuning: Explore the effects of different input resolutions, sequence lengths, and hyperparameters on the models' performance.
Model Evaluation: Evaluate the accuracy and performance of both TimeSformer variants on the HMDB dataset, analyzing their strengths and weaknesses.
Results and Findings
The experiments conducted with the TimeSformer model on the HMDB dataset revealed the following key findings:

Importance of Hyperparameter Tuning
Impact of Distributed Learning on Class Distribution
Considerations for Addressing Class Imbalance
Contributors
Jeevanatham
Azhageswari
Feel free to contribute, suggest improvements, or report issues by forking the repository, making changes, and creating a pull request. For any queries or suggestions, contact [Your Contact Information].
