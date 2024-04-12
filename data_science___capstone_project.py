
"""Data Science | Capstone Project on AIHealth | By -Rohini vishal Raghuwanshi.ipynb



## **Objective:**

- Provide the best performing model to determine probability for a patient having a heart disease or attack.
- Providing the most important drivers for a heart disease or attack.

## **Data Dictionary**

The data provided consists of the following Data Dictionary:

- HeartDiseaseorAttack: Target variable determining whether patient had prior heart disease or heart attack.
- HighBP: Binary flag determining whether a patient has high blood pressure.
- HighChol: Binary flag determining whether a patient has high cholesterol levels.
- BMI: Numeric value representing the Body Mass Index.
- Smoker: Binary flag determining whether a patient smokes or not.
- Diabetes: Binary flag determining whether a patient has diabetes or not.
- Fruits: Binary flag determining whether a patient consumes fruits in daily diet or not.
- Veggies : Binary flag determining whether a patient consumes vegetables in daily diet or not.
- HvyAlcoholConsump: Binary flag determining whether a patient is a heavy consumer of alcohol.
- MentHlth: Numeric value representing mental fitness, ranging from 0 to 30.
- PhysHlth: Numeric value representing physical fitness, ranging from 0 to 30
- Sex: Determining gender of the patient
- Age: The age of the patient binned into buckets between 1-13
- Education: The education level of the patient binned into buckets between 1-6.
- Income: The income of the patient binned into buckets between 1-8

## Importing Libraries

- We will start by importing necessary libraries

#**Capstone Project on AIHealth Heart Disease Prediction**

"""

# Step 1 - Importing all the libraries required in this project & ignoring the warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# This code removes limit for the number of displayed columns in a DataFrame. "None" means there is no maximum limit imposed on the columns
pd.set_option("display.max_columns", None)
# This code sets the limit for the number of rows that can be displayed in a DataFrame to 200
pd.set_option("display.max_rows",200)

"""## Loading the Dataset into pandas DataFrame"""

heart_df = pd.read_csv("Kaggle.HeartDisease.csv")
print(heart_df.shape)
heart_df.head()

"""## EDA - Exploratory Data Analysis"""

heart_df.isnull().sum() # There are no null values in the dataset

# We can also check using the info() where we get the count of Non-Null values and DataTypes
heart_df.info()

# Now, let's check the descriptive statistics for the numerical columns(It gives the summary of Central Tendency, Dispersion and Shape of the distribution of the numerical
# data in the DataFrame)

heart_df.describe()

# To check the unique values in the features
a = 0
for i in heart_df:
  print(heart_df.columns[a], heart_df[i].unique(),'\n')
  a+=1

# Lets check if the data is balanced or unbalancd in the Target column

hrt_d = (heart_df['HeartDiseaseorAttack'] == 1.0).sum()
no_hrt_d = (heart_df['HeartDiseaseorAttack'] == 0.0).sum()

sizes = [hrt_d, no_hrt_d]
labels = ['Heart Disease', 'No Heart Disease']

plt.figure(figsize=(8,4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle = 10)
plt.title('Heart Disease Rate')
plt.axis('equal')
plt.show()

"""- The target variable is unbalanced.
- Looking at this, we will use metrics like precision and recall to evaluate the model
"""

# Lets check the Correlation amongst the variables
plt.figure(figsize=(13,6))
sns.heatmap(heart_df.corr(), annot=True, cmap='Greens')
plt.title('Correlation Matrix')
plt.show()

# Let's check the Outliers for the columns that doesn't have binary values like [BMI, Diabetes, MentHlth, PhysHlth, Age, Education, Income]

# Binary value variables: [HeartDiseaseorAttack [0. 1.], HighBP [1. 0.], HighChol [1. 0.], Smoker [1. 0.], PhysActivity [0. 1.], Fruits [0. 1.], Veggies [1. 0.], HvyAlcoholConsump [0. 1.], Sex [0. 1.]

non_binary_col = ['BMI', 'Diabetes', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

Q1 = heart_df[non_binary_col].quantile(.25)
print("The Q1 is:",'\n', Q1,'\n')

Q3 = heart_df[non_binary_col].quantile(.75)
print("The Q1 is:",'\n',Q3,'\n')

IQR = Q3 - Q1
print("The IQR is:",'\n', IQR,'\n')

lcf = Q1 - 1.5*IQR
ucf = Q3 + 1.5*IQR

print("The LCF is:", lcf,'\n')

print("The UCF is:", ucf,'\n')

# Let's visualize the Outliers using BoxPlot

plt.figure(figsize=(15,5))
sns.boxplot(heart_df)
plt.title("Data Distribution")
plt.xticks(heart_df.columns, rotation=90)
plt.show()

# Univariate analysis to understand the distribution of features
# Define the key features
key_features = ['BMI', 'MentHlth', 'PhysHlth']

# Plot the distribution of each feature w.r.t. Heart Disease classes
for feature in key_features:
    plt.figure(figsize=(7, 5))
    sns.histplot(data=heart_df, x=feature, hue='HeartDiseaseorAttack', kde=True)
    plt.title(f'Distribution of {feature} with respect to Heart Disease')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend(title='Heart Disease', labels=['No Heart Disease', 'Heart Disease'])
    plt.show()

# Multivariate analysis to determine the correlations and analysis of target variables using 3 different colours:
def multivariate_analysis(data):
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

multivariate_analysis(heart_df)

# To determine if new features can be created, based on the given data.

# Feature Engineering
# In this example, let's create a new feature 'TotalFitness' by summing 'MentHlth' and 'PhysHlth'
heart_df['TotalFitness'] = heart_df['MentHlth'] + heart_df['PhysHlth']

# Let's visualize the distribution of the new feature
plt.figure(figsize=(8, 6))
sns.histplot(data=heart_df, x='TotalFitness', kde=True)
plt.title('Distribution of Total Fitness')
plt.xlabel('Total Fitness')
plt.ylabel('Frequency')
plt.show()

"""## Data Preprocessing

- After exploring the dataset, I found that a few of the variables need to be converted from categorical variables to dummy variables using One Hot Encoding and scale the values before training the Machine Learning models. Let's use get_dummies to create dummy columns for the categorical variables.
"""

# Categorical Variables - [Age, Education, Income] - One Hot Encoding

heart_df = pd.get_dummies(heart_df, columns = ['Age', 'Education', 'Income'])
heart_df.head()

# Now let's scale the variables | Scaling - ['BMI', 'MentHlth', 'PhysHlth']
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['BMI', 'MentHlth', 'PhysHlth', 'TotalFitness']

heart_df[columns_to_scale] = standardScaler.fit_transform(heart_df[columns_to_scale])

heart_df.head()

# Let's see the data distribution after scaling

plt.figure(figsize=(15,5))
sns.boxplot(heart_df)
plt.title("Data Distribution after scaling the variables")
plt.xticks(heart_df.columns, rotation=90)
plt.show()

"""# Layout binary classification experimentation space - Model Building

## Training Models

- Logistics Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
"""

# Machine Learning: Splitting our heart_df dataset into Training & Testing

#Splitting the Dataset
from sklearn.model_selection import train_test_split
Y = heart_df['HeartDiseaseorAttack']
X = heart_df.drop(['HeartDiseaseorAttack'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Handling Class Imbalance using SMOTE (Synthetic Minority Over-Sampling Technique) and RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
sm = SMOTE(sampling_strategy=1, random_state=10, k_neighbors=5)
X_train_smote, Y_train_smote = sm.fit_resample(X_train, Y_train)

rus = RandomUnderSampler(random_state=1)
X_train_rus, Y_train_rus = rus.fit_resample(X_train, Y_train)

# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

#Initialize the models
logistic_regression_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()
gradient_boosting_model = GradientBoostingClassifier()
xgboost_model = xgb.XGBClassifier()

#Train the models
logistic_regression_model.fit(X_train_smote, Y_train_smote)
decision_tree_model.fit(X_train_rus, Y_train_rus)
random_forest_model.fit(X_train_smote, Y_train_smote)
gradient_boosting_model.fit(X_train_smote, Y_train_smote)
xgboost_model.fit(X_train_smote, Y_train_smote)

"""## Model Evaluation and Visualization"""

# Precision Recall Curve for the best threshold
from sklearn.metrics import precision_recall_curve
def plot_precision_recall_curve(model, X_test, Y_test):
  y_scores = model.predict_proba(X_test)[:,1]
  precision, recall, thresholds = precision_recall_curve(Y_test, y_scores)
  plt.plot(recall, precision, marker='.')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve')
  plt.show()

# Plotting Precision-Recall Curve for each model
for name, model in [('Logistic Regression',logistic_regression_model),
                    ('Decision Tree', decision_tree_model),
                    ('Random Forest', random_forest_model),
                    ('Gradient Boosting', gradient_boosting_model),
                    ('XGBoosting', xgboost_model)]:
    print(f"Precision-Recall Curve for {name}:")
    plot_precision_recall_curve(model, X_test, Y_test)

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
def evaluate_model(model, X_test, Y_test):
  y_pred = model.predict(X_test)
  print(f"Model Evaluation for {model.__class__.__name__}:",'\n')
  print(classification_report(Y_test, y_pred))
  print("Confusion Matrix:")
  print(confusion_matrix(Y_test, y_pred),'\n')
  print("===============================================================")

# Evaluate each model
for name, model in [('Logistic Regression', logistic_regression_model),
                    ('Decision Tree', decision_tree_model),
                    ('Random Forest', random_forest_model),
                    ('Gradient Boosting', gradient_boosting_model),
                    ('XGBoosting', xgboost_model)]:
    evaluate_model(model, X_test, Y_test)

# Feature Importance
import seaborn as sns
import pandas as pd
def plot_feature_importance(model, feature_name):
  if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({'Feature': feature_name, 'Importance':model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10,6))
    ax = sns.barplot(data=feature_importance, x='Importance', y='Feature')
    ax.bar_label(ax.containers[0], fontsize=8);
    plt.title('Feature Importance for '+ model.__class__.__name__)
    plt.show()

# Plotting feature importance for each model
for name, model in [('Logistics Regression', logistic_regression_model),
                    ('Decision Tree', decision_tree_model),
                    ('Random Forest', random_forest_model),
                    ('Gradient Boosting', gradient_boosting_model),
                    ('XGBoost', xgboost_model)]:
    plot_feature_importance(model, X.columns)

"""## Hyperparameter Tuning and Performance Collection"""

# Hyperparameter tuning and performance collection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score


param_grids = {
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'Decision Tree': {'max_depth': [None, 10, 20, 30, 40, 50]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth':[None, 10, 20, 30, 40, 50]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]}
}

results = []

for name, model in [('Logistic Regression', logistic_regression_model),
                    ('Decision Tree', decision_tree_model),
                    ('Random Forest', random_forest_model),
                    ('Gradient Boosting', gradient_boosting_model),
                    ('XGBoost', xgboost_model)]:
    print(f"Training and evaluating {name} model...")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring='f1', cv=5)
    grid_search.fit(X_train_smote, Y_train_smote)

    best_model = grid_search.best_estimator_

    evaluate_model(best_model, X_test, Y_test)

    y_pred = best_model.predict(X_test)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    results.append({'Model': name, 'Precision': precision, 'Recall': recall, 'F1': f1})

# Create DataFrame of results
results_df = pd.DataFrame(results)
print("\nResults DataFrame:")
print(results_df)

"""## Model Pipeline"""

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Define pipelines for each model
pipelines = {
    'Logistic Regression': ImbPipeline([('scaling', StandardScaler()),
                                         ('sampling', SMOTE(sampling_strategy=1, k_neighbors=5, random_state=10)),
                                         ('model', LogisticRegression())]),
    'Decision Tree': ImbPipeline([('scaling', StandardScaler()),
                                   ('sampling', RandomUnderSampler(random_state=1)),
                                   ('model', DecisionTreeClassifier())]),
    'Random Forest': ImbPipeline([('scaling', StandardScaler()),
                                   ('sampling', SMOTE(sampling_strategy=1, k_neighbors=5, random_state=10)),
                                   ('model', RandomForestClassifier())]),
    'Gradient Boosting': ImbPipeline([('scaling', StandardScaler()),
                                      ('sampling', SMOTE(sampling_strategy=1, k_neighbors=5, random_state=10)),
                                      ('model', GradientBoostingClassifier())]),
    'XGBoost': ImbPipeline([('scaling', StandardScaler()),
                             ('sampling', SMOTE(sampling_strategy=1, k_neighbors=5, random_state=10)),
                             ('model', xgb.XGBClassifier())])
}

# Train and evaluate models using pipelines
for name, pipeline in pipelines.items():
    print(f"\nTraining and evaluating {name} using pipeline...")
    pipeline.fit(X_train, Y_train)
    evaluate_model(pipeline, X_test, Y_test)

# Visualising the Best Performin Model

# Import necessary libraries
import matplotlib.pyplot as plt

# Step 1: Analyze the results to identify the best-performing model based on Precision, Recall, and F1-score.
best_model = results_df.loc[results_df['F1'].idxmax()]
print("Best Performing Model:")
print(best_model)

# Step 2: Visualize the performance metrics to compare different models.
plt.figure(figsize=(10, 6))

plt.plot(results_df['Model'], results_df['Precision'], marker='o', label='Precision')
plt.plot(results_df['Model'], results_df['Recall'], marker='o', label='Recall')
plt.plot(results_df['Model'], results_df['F1'], marker='o', label='F1-score')

plt.title('Performance Metrics Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
