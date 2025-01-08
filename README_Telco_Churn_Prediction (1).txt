
README: Telco Customer Churn Prediction and Model Comparison

This README explains the steps we followed to preprocess the dataset, train multiple machine learning models, evaluate their performance, and identify important features. Each step is broken down for simplicity.

1. Loading the Dataset
- What we did:
  - We used the file `Telco_customer_churn.xlsx` to load customer data into a pandas DataFrame.
- Why we did it:
  - To prepare and analyze the data for predicting customer churn.

2. Preprocessing the Data
- Dropping Unnecessary Columns:
  - We removed columns like `CustomerID`, `Country`, `State`, and others that didn’t contribute to prediction.
  - Reason: These columns don’t directly affect customer churn.

- Converting the Target Column:
  - We converted the `Churn Label` column to numeric values (1 for "Yes," 0 for "No").
  - Reason: Machine learning models work with numbers, not text.

- Handling Missing Values:
  - We converted `Total Charges` to numeric and filled missing values with `0`.
  - Reason: Models can't handle empty cells.

- Encoding Categorical Data:
  - For binary columns (e.g., `Gender`, `Partner`), we replaced text with numeric values (0 or 1).
  - For multi-class columns (e.g., `Internet Service`, `Payment Method`), we used one-hot encoding to create separate columns for each category.
  - Reason: Machine learning models require numbers, not categories.

3. Balancing the Data
- What we did:
  - Used SMOTE (Synthetic Minority Oversampling Technique) to balance churn (Yes/No) counts.
- Reason: Balanced data ensures the model doesn’t favor the majority class.


4. Feature Scaling
- What we did:
  - Scaled the features using StandardScaler to standardize the values.
- Reason: Models like Logistic Regression and SVM perform better when features are on the same scale.

5. Training-Test Split
- What we did:
  - Split the data into training (70%) and testing (30%) sets.
  - Reason: The model needs separate data for training and evaluation to avoid overfitting.

6. Training and Evaluating Models
We trained and tested six models:

1. Logistic Regression
   - Simple and interpretable.
   - Suitable for binary classification problems.

2. Random Forest
   - A tree-based model that combines multiple decision trees.
   - Captures non-linear relationships in the data.

3. Decision Tree
   - A single tree-based model.
   - Easy to visualize but prone to overfitting.

4. XGBoost
   - An advanced boosting algorithm that works well with structured data.
   - Powerful for feature importance and accuracy.

5. LightGBM
   - A boosting algorithm optimized for speed and memory.
   - Works well with large datasets.

6. SVM (Linear Kernel)
   - Finds the best boundary between churn and non-churn customers.

7. Model Comparison
- What we did:
  - For each model, we calculated:
    - Accuracy: Percentage of correct predictions.
    - AUC-ROC: Measures the model’s ability to distinguish between classes.
    - Precision, Recall, F1-Score: Metrics to evaluate performance on predicting "Churn" customers.
  - Saved the results into an Excel file for comparison.
  - Reason: To identify the best-performing model.

8. Identifying Important Features
- Feature Importance Extraction:
  - For Random Forest and XGBoost, we extracted feature importance scores.
  - For Logistic Regression, we used the coefficients of the model to identify important features.
  - Reason: To understand which features impact customer churn the most.

- Recursive Feature Elimination (RFE):
  - Used RFE to select the top 10 features for Logistic Regression.
  - Reason: To simplify the model without losing accuracy.

- SHAP (SHapley Additive exPlanations):
  - Visualized XGBoost’s feature importance using SHAP.
  - Reason: To explain the predictions made by the model.

9. Final Features
- What we did:
  - Combined the top 10 features identified from all models.
  - Reason: To focus on the most impactful variables for future models.

10. Results
- The model comparison results were saved as `model_comparison_results.xlsx`.
- Key takeaways include:
  - Best Model: The model with the highest AUC-ROC score.
  - Important Features: Features that contribute the most to churn prediction.

Instructions to Run
1. Ensure the dataset `Telco_customer_churn.xlsx` is in the working directory.
2. Install required Python libraries:
   ```
   pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap
   ```
3. Run the script in Python or Google Colab.
4. Review the output:
   - Feature importance and SHAP plots.
   - Model comparison results.

This documentation explains each step in simple terms, making it easy for anyone to reproduce the churn prediction tasks and understand the workflow.
