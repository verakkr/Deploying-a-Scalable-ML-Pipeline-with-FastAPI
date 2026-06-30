# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf


## Model Details
- **Developed by:** Kristopher Verakul for WGU Course D501 Machine Learning DevOps and Udacity Machine Learning Dev Ops
- **Model date:** June 2026
- **Model Version:** v 1.0
- **Model Type:** Random Forest Classifier for binary classification 
- **Target Variable:** salary
- **Prediction Output:** Greater than or less than 50k
- **Artifacts:** Trained model saved to \model folder


## Intended Use
The model is intended to train a user on how to evaluate, load, save, and deploy a model using a repeatable pipeline. In this specific use case, we will be evaluating a US Census dataset.


## Training Data
The model is trained on US Census bureau data which is provided as part of this course. The census dataset contains 32561 rows and we will be targeting the 'Salary' column. Additional columns in the dataset include the following:
- Age
- workclass (federal, state, local government, private, etc...)
- fnlgt
- education
- education-num (highest grade level)
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country
- salary


## Evaluation Data
The results of the data came from a test split of the original Census dataset. The test set was not used to train the model and was used to evaluate the performance of the model across categorical slices. We evaluated performance across several categorical variables including education, marital status, workclass, occupation, relationship, race, sex, and native country. 


## Metrics
The model was evaluated using precision, recall, and F1 score. 

Results:
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1:** 0.6863

With a precision score of 0.7419 the model was right about 74% of the time when predicting people earning greater than 50k. The recall of 0.6384 means that the model only found about 64% of people earning greater than 50k. 

## Caveats and Recommendations
This model was created and used for a class project and should not be used on real world dataset. The dataset the model was trained on was a dataset that was provided as part of the course, and could reflect inaccurate data. 