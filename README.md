# E-Governance Complaint Priority Prediction

## Week 4 - Model Development and Evaluation

This project demonstrates a machine-learning workflow for an e-governance/digital public-service use case: **predicting the priority of citizen complaints** as Low, Medium, or High.

> **Dataset note:** `data.csv` is a synthetic dataset created for internship/academic demonstration. It does not contain real citizen data and should not be used for production decisions without validation, governance review, bias testing, privacy controls, and an approved real-world dataset.

## Objective
Build, train and evaluate a classification model that can provide an initial priority signal for incoming public-service complaints.

## Workflow
1. Create/inspect the dataset.
2. Separate features and target.
3. Encode categorical data.
4. Split data into training and testing sets.
5. Train a Random Forest classifier.
6. Generate predictions.
7. Evaluate with accuracy, precision, recall and F1-score.
8. Visualize the confusion matrix, metrics and feature importance.

## Model
**Random Forest Classifier** was selected because it can model non-linear relationships, works with mixed feature types after preprocessing, and provides a practical baseline for tabular classification.

## Evaluation result (20% test split)
- Accuracy: **0.685**
- Weighted Precision: **0.684**
- Weighted Recall: **0.685**
- Weighted F1 Score: **0.681**

These values are only for the synthetic demonstration dataset and are not evidence of real-world performance.

## Files
- `data.csv` - synthetic complaint dataset
- `train_model.py` - preprocessing, training and evaluation
- `predict.py` - example prediction for a new complaint
- `complaint_priority_model.joblib` - trained model artifact
- `requirements.txt` - Python dependencies
- `plots/` - evaluation graphics

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Digital Governance Considerations
The model should be treated as decision support, not as an automatic final decision-maker. Real deployment would require human review, privacy protection, audit logs, monitoring for bias, explainability, secure data handling and periodic model validation.
