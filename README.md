****Overview****

This repository contains a project focused on stress testing in financial institutions. 

This initiative explores how financial institutions can leverage internal financial indicators for effective stress testing. Using historical U.S. economic and financial data from FRED (January 1990 to July 2024), I developed a classification model designed to predict systemic stress events based on historical trends and patterns. Additionally, I incorporated Monte Carlo simulations to evaluate outcomes under various scenarios. This project demonstrates how financial institutions can enhance preparedness and decision-making through data-driven insights.

**The repository includes:**
  - Jupyter Notebooks for data preparation, model building, and simulations.
  - A processed dataset ready for modeling.
  - Scripts for helper, visualization, evaluation and customized classifier functions.

**Aim of the Project**
 - The project aims to:
    1. Predict systemic stress events using financial indicators derived from historical data.
    2. Utilize Monte Carlo simulations to evaluate model robustness under various scenarios.
    3. Provide actionable insights through well-documented workflows for data preparation, feature engineering, and model evaluation.
  
**Project Structure**
  
  *Jupyter Notebooks:*
  -  stress_testing_dataset_preparation.ipynb
     Purpose: Prepares the dataset for modeling by performing cleaning, feature engineering, and transformations.
  - stress_testing_classification_model.ipynb
     Purpose: Builds and evaluates a classification model for identifying stress events.
  -  stress_testing_monte_carlo_simulation.ipynb
     Purpose: Applies Monte Carlo simulations to assess the impact of feature variability on model predictions.
  
  *Dataset:*
  - data_ready_for_model.csv: Contains the processed dataset used for training and testing the classification model. Includes financial features, derived indicators, and stress labels.
  
  *Plots and Scripts*
  - Plots:
    Visualizations created in the notebooks, such as feature distributions, confusion matrices, ROC curves, and probability distributions.
    These plots provide insights into model performance and feature behavior.
  - Helper Functions:
    Common functions used across notebooks, such as data cleaning, transformation utilities, and model evaluation metrics.
    Functions are saved as Python scripts for reusability.
