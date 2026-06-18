# PRML: Comparative Study of KNN and Weighted KNN

This project presents a comparative analysis of standard K-Nearest Neighbors (KNN) and inverse-distance-weighted KNN using the Palmer Penguins dataset. The study investigates whether distance-weighted voting improves classification accuracy and under what conditions such improvements manifest.

## Project Overview
The core objective is to perform a multi-class classification to predict the species of penguins (Adelie, Chinstrap, or Gentoo) based on four continuous morphological measurements: bill length, bill width, flipper length, and body mass.

## Research Questions
This study addresses the following key questions:
* Does Weighted KNN outperform classic KNN on this dataset?
* What is the optimal $k$ for each method?
* How much does feature standardization affect the results?
* Where do the two methods disagree, and what does the confusion matrix reveal about the underlying class geometry?

## Deliverables
* **Full Project Report**: Covers the introduction, dataset description, EDA, experimental setup, results, and conclusions.
* **Jupyter Notebook**: Contains all from-scratch implementations, visualizations, decision boundary mappings, and confusion matrices.
