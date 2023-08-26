# ***Import libraries and the dataset***

# Install necessary packages
!pip install optuna
!pip install shap
!pip install catboost
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, ensemble, model_selection, tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay, fbeta_score, make_scorer
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier, Pool, cv
import optuna
import shap
warnings.filterwarnings('ignore')

# Install kaggle API
!pip install kaggle

# Create .kaggle directory if it doesn't exist
!mkdir -p ~/.kaggle

# Place kaggle API keys in the directory
!echo '{"username":"hsleecri","key":"26fd0d550bb5b6e60a50abd35f9f46b9"}' > ~/.kaggle/kaggle.json

# Change the permissions of the file
!chmod 600 ~/.kaggle/kaggle.json

#파일 디렉토리 설정
!mkdir data
%cd data

#분석 대상 데이터 다운
!kaggle datasets download -d subham07/detecting-anomalies-in-water-manufacturing

#압축 해제
!unzip detecting-anomalies-in-water-manufacturing.zip

# CSV 파일 읽기
data = pd.read_csv("Train.csv")

# ***Preprocess the data***

# Separate the input features (X) and the target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Find features with zero variance
zero_var_features = X.columns[X.var() == 0]
num_zero_var_features = len(zero_var_features)

# Exclude features with zero variance
X = X.drop(zero_var_features, axis=1)

# Print features with zero variance
print("Features with zero variance:")
print(zero_var_features)
print("Number of features with zero variance:", num_zero_var_features)

# Train a random forest classifier on the scaled features
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_

# Sort feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]

# Select top k important features
k = 8
selected_features = X.columns[sorted_indices[:k]]

# Subset the data with selected features
X_selected = X[selected_features]

# Print selected features and their importances
print("Selected features and importances:")
for feature, importance in zip(selected_features, feature_importances[sorted_indices[:k]]):
    print(feature, ":", importance)



# Specify your categorical features
cat_features = list(X_selected.columns[3:])
print(cat_features)

# Convert X_selected to pandas DataFrame
X_selected = pd.DataFrame(X_selected, columns=selected_features)

# Print information about the selected DataFrame
print("Selected DataFrame shape:", X_selected.shape)
print("Selected DataFrame head:")
print(X_selected.head())

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

from imblearn.over_sampling import BorderlineSMOTE
smote = BorderlineSMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ***fit the catboost model with optimizing the hyperparmeters by using Optuna***

def objective(trial):
    params = {
        'iterations' : trial.suggest_int('iterations', 50, 300),
        'depth' : trial.suggest_int('depth', 4, 10),
        'learning_rate' : trial.suggest_uniform('learning_rate', 0.01, 0.3),
        'random_strength' : trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature' : trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait' : trial.suggest_int('od_wait', 10, 50)
    }
    
    model = CatBoostClassifier(**params, loss_function='Logloss', eval_metric='AUC', nan_mode='Min', 
                               leaf_estimation_iterations=10, use_best_model=True)
    
    tss = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_index, valid_index in tss.split(X_train_res):
        X_train_fold, X_valid_fold = X_train_res.iloc[train_index, :], X_train_res.iloc[valid_index, :]
        y_train_fold, y_valid_fold = y_train_res.iloc[train_index], y_train_res.iloc[valid_index]

        train_dataset = Pool(data=X_train_fold, label=y_train_fold)
        eval_dataset = Pool(data=X_valid_fold, label=y_valid_fold)

        model.fit(train_dataset, eval_set=eval_dataset, early_stopping_rounds=100, verbose=False)
        
        preds = model.predict(X_valid_fold)
        score = fbeta_score(y_valid_fold, preds, beta=0.5)
        scores.append(score)
    
    cv_score = np.mean(scores)
    
    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# Visualize optimization history
optuna.visualization.plot_optimization_history(study)

# Visualize parameter importances
optuna.visualization.plot_param_importances(study)

# ***Evaluation on the test set***

# Get the best parameters
best_params = study.best_trial.params

# Create the model with the best parameters
model = CatBoostClassifier(**best_params, loss_function='Logloss', eval_metric='AUC', nan_mode='Min', 
                           leaf_estimation_iterations=10, use_best_model=True)

# Create Pool data for CatBoost
train_dataset = Pool(data=X_train_res, label=y_train_res)

# Fit the model on the whole training set
model.fit(train_dataset, verbose=False)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Compute the F-beta score
score = fbeta_score(y_test, y_pred, beta=0.5)

print(f'F-beta score on the test set: {score:.4f}')