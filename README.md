**Title of Project**: Women-Cloth-Reviews-Prediction-with-Multinomial-Naive-Bayes

This project analyzes customer sentiment in women’s clothing reviews using the Multinomial Naive Bayes algorithm to classify reviews as positive, negative, or neutral. By understanding sentiment patterns, this model provides insights to enhance customer experience and guide product improvements.

**objective**: Sentiment Prediction
Topic Detection
Model Optimization
Insight Generation
Automated Review Analysis

**Data Source**: Kaggle,In-house Data,Public Review Platforms

**Import Data**:Load the dataset and display initial rows to understand the structure.
**Describe Dtat**:Use descriptive statistics to get insights into the data (e.g., `.describe()`, `.info()`).
- Identify data types, missing values, and summary statistics.
- **Data Visualization**:Visualize the data to understand relationships and distributions (e.g., histograms, scatter plots, heatmaps).
- Use insights to guide data preprocessing.
- **Data Preprocessing**: Handle missing values, encode categorical variables, scale features, or perform any necessary data cleaning.
- Prepare the data to be in a format suitable for modeling.
- **Train Test Split** :Split the data into training and test sets to evaluate model performance later.
- **Modeling** :Choose and train various models based on the problem (classification, regression, etc.).
- Tune hyperparameters if needed for optimal performance.
- **Model Evaluation** :Evaluate the models using appropriate metrics (e.g., accuracy, precision, recall, R², MAE).
- Compare models and select the best-performing model.
- **Prediction**:Use the model to make predictions on the test data.
- If applicable, predict on unseen data or create a submission file.
- **Explaination** :Interpret model results and explain important features or insights.
- Use tools like SHAP or LIME for model interpretability if needed.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('https://raw.githubusercontent.com/YBIFoundation/ProjectHub-MachineLearning/refs/heads/main/Women%20Clothing%20E-Commerce%20Review.csv')
df.head()
df.info()
df.shape
#Missing values
df.isna().sum()
df[df['Review']==""]=np.NaN
df['Review'].fillna("No Review",inplace=True)
df.isna().sum()
df['Review']
df.columns
x=df['Review']
y=df['Rating']
df['Rating'].value_counts()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,stratify=y,random_state=2529)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(lowercase=True,analyzer='word',ngram_range=(2,3),stop_words='english',max_features=5000)
x_train=cv.fit_transform(x_train)
cv.get_feature_names_out()
x_train.toarray()
x_test=cv.fit_transform(x_test)
cv.get_feature_names_out()
x_test.toarray()
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred.shape
y_pred
model.predict_proba(x_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
df['Rating'].value_counts()
df.replace({'Rating':{1:0,2:0,3:0,4:1,5:1}},inplace=True)
y=df['Rating']
x=df['Review']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,stratify=y,random_state=2529)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(lowercase=True,analyzer='word',ngram_range=(2,3),stop_words='english',max_features=5000)
x_train=cv.fit_transform(x_train)
x_test=cv.fit_transform(x_test)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred.shape
y_pred
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
