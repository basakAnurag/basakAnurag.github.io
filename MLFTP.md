# Data Processing and its need

Convert raw data to clean data which the models understand
Used to: Handle the missing values, duplicate values, anomalies, categorical values etc.

# Hyperparameter tuning and its need

To find the best parameters, and improve model performance.

# Difference between GridSearchCV and RandomizedSearchCV. What is CV?

GridSearchCV goes through all possible combinations. Slower.
RandomizedSearchCV randomly chooses hyperparameters and checks them. Faster.
Cross-Validation is a technique where the unseen data is divided into k folds. The model trains on k-1 folds and tests on the kth fold. This occurs for n iterations.

# Need for Scaling. Different types of Scaling.

Scaling is used in datasets to keep the numerical values comparable so that one with large values do not dominate the other with small values.

Scaling Techniques:

StandardScaler: z = (x-mean)/S.D

MinMaxScaler:   z = (x-min)/(max-min) | Range [0,1]

RobustScaler:   z = (x-Q2)/(Q3-Q1)

# PCA

Dimensionality reduction do not reduce data-points.It reduces number of columns but keeping the rows intact.
Eg: (1000, 50) -> (1000, 10)
PCA stands for Principle Component Analysis. The new components capture the maximum variance of the data. The second captures the next highest variance while being orthogonal to the first.
Note that some variance is lost due to some discarded components.

# Encoders

Encoding is the technique to convert categorical variables to numerical.

1. LabelEncoding: Assign a unique integer to each category. Useful for Ordinal data only (not nominal data)
2. OneHotEncoding: Assign a binary colunmn to each category.
3. OrdinalEncoding: Assigns number in a meaningful order.
4. TargetEncoding: Replaces each category with the average value of the TargetValue for that category.
5. BinaryEncoding: Converts categories to integers in binary format. Fewer columns than OneHotEncoding.
6. FrequencyEncoding: Replaces each category with the frequency of the category.

# Solver

---


# Bias vs Variance

Bias = Accuracy -> Detects underfitting | Model is too simple.

Eg: Model is y = x^2 but is trained using y = mx + c (This is high bias)

Variance = Stability -> Detects overfitting | Model is too complex

Eg: Predictions change drastically where only few points change (This is high variance)

Thus, low variance and low bias is required.

# Stratify

Stratify ensures that class distribution remains same in both train and test datasets after splitting.

Eg: 80 of class A and 20 of class B

stratify=y ensures equal distribution for both the classes. Eg: (20+5) instead of (25 + 0)

# Correlation between categorical variables

We canNOT use Pearson correlation directly on Categorical variables.

Methods:

1. Chi-square Test - Whether 2 categorical variables are associated.
2. Cramer's V: Strength of association between 2 vairables.
3. Phi coefficient: Used when both variables have only 2 categories.

# RFE (Recursive Feature Elimination)

Recursively removes the least important features unless number of desirable features remain.

Thus, reduces overfitting, removes irrelevant features, makes models simpler to interpret.

```Python
import sklearn

model = RandomForestRegressor(random_state=42)
rfe = RFE(
  estimator=model,
  n_features_to_select=5
)

X = rfe.fit_transform(X_train, y_train)
```

# Decision Trees

A ML algorithm which makes predictions by splitting the data into smaller groups based on categories.

Types:

Classification Trees - Binary groups (Yes/No) -> Uses: Gini Impurity, Entropy gain

Regression Trees - Predicts values (Age group) -> Uses: MAE, MSE

Advantages: No FeatureScaling required, Captures Non-linear relationships

Disadvantages: Tends to overfit, Sensitive to small changes in data

# K means clustering vs KNN (K nearest neighbours)

|  K means clustering   |            KNN            |
| :-------------------: | :-----------------------: |
| Unsupervised learning |    Supervised learning    |
|    Finds clusters     |      Predicts labels      |
|  No target variable   | Requires target variables |

**K means** tries to minimise Within Sum of clusters (WCSS)
Squaring ensure -> No negative values and large distances penalised more. thus can be optimised
**K means++** differs from K means only on the initialisation of clusters. Instead of random clustering, it initialises one cluster at a point, then the other at a far-away point ...

```Python
import sklearn

km = KMeans(
  n_means = 2,
  random_state=42
)
km.fit(X)
print(km.labels_)
```

# Elbow-method and silhouette score

**NOTE: Elbow-method is subjective whereas Silhouette score is objective**

Finds the optimum number of clusters

```Python
import sklearn

wcss = []
for i in range(1, 11):
  model = KMeans(n_clusters=i, random_state=42)
  model.fit(X)
  wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.show()
```

inertia_ calculates WCSS(Within clusters sum of squares)

Silhouette score (S) = (b-a)/max(a, b)

where, 

a = average distance to points inside its own cluster

b = average distance to points to nearest cluster

Range: [-1, 1]

`1 -> Perfect clustering` 

`0 -> Clusters overlap` 

`-1 -> Wrong clustering`

```Python
import sklearn

for k in range(2,11):
    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={k}, Score={score:.3f}")
```


# Pipeline + SVC

```Python
import sklearn

pipeline = Pipeline([
  ('imputer', SimpleImputer())
  ('model', SVC(random_state=42))
])

pipeline.fit(X_train, y_train)
preds = piepline.predict(X_test)
```

C in SVC stands for classifier.

Note: small C means simpler model wide margin or allows some mistakes (lower risk of overfitting) whereas large C means complex model, narrow margin (higher risk of overfitting)

SVM can classify non-linear data using polynomial, sigmoid, RBF(radial basis function) functions.

Scaling is required in SVM because SVM uses distance parameter to find the optimal hyperplane. Features of larger values can dominate in the distance calculation.

|  Logistic Regression   |                   SVM                   |
| :--------------------: | :-------------------------------------: |
| Predicts probabilities |      Finds the optimal hyper-plane      |
| Uses sigmoid function  | Uses kernel functions to transform data |

# Activation Function

Whether a neuron should produce output and introduce non-linearity into a neural network. Without it a neural network is a linear model.

Common activation functions:

- ReLu: f(x) = max(0,x)
- Sigmoid: Value between 0 and 1
- Tanh: Value between -1 and 1

# Neural networks

A mesh like model inspired from human brain where connecting neurons learns and predicts output from patterns of data.

Input layer -> Hidden layers -> Output layer

---

---

# Recent questions (starting from `7/18/2026`)

# K - Fold cross validation

is a technique to check how our model performs on unseen data.

Why we need it? If the unseen data is easy, then our model overperforms, but underperform if the unseen dataset is hard.

We split the sample into k folds, we train on k-1 folds and test on kth fold and this occurs k times. We then take the average.

```Python
import sklearn

model = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
  model,
  X,
  y,
  cv=kfold,
  scoring="root_mean_square_error"
)
print(scores)
print(scores.mean())
```

# GridSearchCV vs RandomizedSearchCV

```Python
import sklearn

model = Ridge()
params = {
  "alpha" : [0.1, 1, 10]
}
grid = GridSearchCV(
  estimator=Ridge(),
  param_grid=params,
  cv=5,
  scoring='root_mean_square_error'
)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

```Python
import sklearn

model = LinearRegression()
param_grid = {
  'alpha': [0.1, 1, 10]
}
grid = RandomizedSearchCV(
  model,
  X,
  y,
  param_grid=param_grid
  cv=5,
  random_state=42
)

grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

# Train a Linear Regression model

```Python
import sklearn
import pandas as pd

X = train.drop(["TargetValue"], axis=1)
y = train["TargetValue"]

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print(mse)
```

# RandomForestRegressor

RandomForest is an algorithm which builds many trees and predicts the output by taking the output of the trees. Each tree is built randomly on subset of data and subset of features. This ensures the trees are independent of each other. In regression the output is the average of all the trees and in Classifier it is the majority voting among the trees. This reduces overfitting and improves accuracy.

# OneHotEncoder

Creates binary columns for existing columns to covert categorical data to numerical data.
Eg:

|   A   |     | A_Red | A_Blue | A_Green |
| :---: | --- | :---: | :----: | :-----: |
|  Red  |     |   1   |   0    |    0    |
| Blue  |     |   0   |   1    |    0    |
| Green |     |   0   |   0    |    1    |

```Python
import sklearn

categorical_cols = ["A", "B"]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
preprocessor = ColumnTransformer(transformer=[
  ("cat", OneHotEncoder(random_state=42, sparse_output=False, handle_unknown='ignore'), categorical_cols)
],
remainder='passthrough')
X_preprocessed_train = preprocessor.fit_transform(X_train)
X_preprocessed_test = preprocessor.transofrm(X_test)
```

# PCA

```Python
import sklearn

df = pd.read_csv(...)
X = df.select_dtypes(include=['int64', 'float64'])
ss = StandardScaler()
X_scaled = ss.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)
print(pca.components_)
```

# Plotting in matplotlib.pyplot

```Python
import sklearn
import matplotlib.pyplot as plt

points = [0.910, 0.901, 0.902]
label = [Cat, Light, X]
plt.bar(label, points)
plt.xlabel("Name")
plt.ylabel("R2")
plt.title("Graph")
plt.legend()
plt.show()
```

# Load iris dataset

```Python
import sklearn

iris = load_iris()
X = iris.data
y = iris.target
```

# Apply preprocessing on features of X_train and X_test dataset

**For Categorical Features**
Apply OneHotEncoding from sklearn library on all categorical features(object columns). Do Encoding in the order of following list

Categorical Features = ['gender', 'area','qualification','marital_status', 'num_policies', 'policy', 'type_of_policy']

Lets call the transformed caterical feature matrix  X1

**For Numerical Features**
apply MinMaxScaler and transform the dataset. Do scaling in the order of following list:

Numerical Features = [ 'income', 'vintage', 'claim_amount' ]

Lets call the transformed numerical feature matrix X2
Concatenate(One Hot Encoded Features, Scaled Numerical Features)
After combining transformed categorical feature( X1 ) matrix and transformed numerical feature matrix ( X2 ) (side by side in that order), the output will be  X = [X1 X2]

```Python
# Solution
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

cat = ['gender', 'area','qualification','marital_status', 'num_policies', 'policy', 'type_of_policy']
num = ['income', 'vintage', 'claim_amount']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
mMS = MinMaxScaler()

preprocessor = ColumnTransformer(transformers=[
    ('ohe', ohe, cat),
    ('mMs', mMs, num)
])
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

pipeline.set_output(transform='pandas') # Important
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
```

# Load iris dataset

```Python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df.head()
```

# SMOTE (Synthetic minority over-sampling technique)

Unlike over-sampling(creating duplicates prone to overfitting), SMOTE creates synthetic data (nearby examples).
Instead of memorizing/over-fitting model learns over a region.

```Python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

NOTE: SMOTE is only applied on train data, never on test data otherwise data-leakage.

# UnderSampling

Unlike OverSampling and SMOTE Undersampling decreases majority class samples (Loses patterns)

```Python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)
```

Eg: From 990 major class sample -> throw away 980 to get 10:10 sample ratio. That's a huge loss

# Class Weights

Instead of changing in the dataset, one can just set the class weights.

```Python
import sklearn

model = LogisticRegression(class_weight='balanced')
model2 = RandomForectClassifier(class_weight='balanced')
```

**class_weight='balanced'** : 

`Weights = (Total Samples)/(Number of classes * Samples in class)`

# Entropy

(Randomness) or uncertainty

A, A, A, A -> Entropy 0
A, B, A, B -> Entropy 1

Entropy = - (Sigma) p * log2(p) | where p denotes probability

**p**       -> Probability of happening (major class dominates)            Eg: A, A, A, B

**log2(p)** -> Rare events carry more information (certain vs uncertain)   Eg: Chance of B is low but carries more information
'-'         -> log of a number (probability) b/w 0 and 1 thus we need a negative sign to make it positive

**Sigma**   -> Multiple classes contribute together

# Information Gain

Before splitting: Entropy = 1
After splitting: Entropy = 0.3

Thus Information Gain = 1 - 0.3 = 0.7

**IG = Entropy(parent) - Weighted Entropy(children)**

NOTE: Why Weighted? Note that every class doesnot afect equally. Bigger child influences more.

# Gini Index

Note that *Entropy* uses logarithm (computationally expensive) Thus Gini removes that

**Gini = 1 - (Sigma)p^2**

# What kind of patterns can unsupervised learning discover?

1. Clustering (Group similar objects)
2. Dimensionality reduction (Reduce features)
3. Association rules (Prducts come together)
4. Anomaly detection (Find unusual observations)

# Confusion Matrix (Accuracy, Precision, Recall, F1) and ROC-AUC

The 4 terms **TP, FP, TN, FN** needs to be understood first.

| Confusion Matrix |       | Model | Prediction |
| :--------------: | :---: | :---: | :--------: |
|                  |       |  No   |    Yes     |
|       Real       |  No   |  TN   |     FP     |
|      Value       |  Yes  |  FN   |     TP     |

### Accuracy 
You have 1,000 transactions: 990 Normal and 10 Fraud
So fraud is only 1%. Suppose the model is completely useless and says: *"Everything is Normal."*

It gets: 990 normal -> correctly predicted and 10 fraud -> missed

So: Accuracy = 99% But the model caught: 0/10=0% of the fraud.

### Precision
Suppose the model says 20 transactions are fraud. Of those 20: 15 really are fraud whereas 5 are actually normal.

So: Predicted Fraud = 20 and Actual Fraud among them = 15

`Precision: TP/(TP+FP)`

### Recall
Suppose there are actually 100 fraudulent transactions. The model catches 90 and misses 10. Then:

`Recall: TP/(TP+FN)`

### F1 Score

The Harmonic Mean between Recall and Precision is F1 Score.
NOTE that High precision often results in Low recall and vice versa. Thus F1 Score balances both.

`1/F1 = 1/Recall + 1/Precision`

### ROC-AUC

A classification model generally outputs a probability for example 90% fraud, 50% fraud etc. Thus threshold needs to be optimised as well which ROC-AUC chooses.

`AUC = 1 -> Perfect separation`

`AUC = 0.5 -> Equivalent to random guessing`

`AUC = 0 -> Model in the wrong direction`

---