# Import necessary libraries
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Define the FeatureEngineering class


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['Dist_to_Hydrolody'] = (
            X_copy['Horizontal_Distance_To_Hydrology']**2 + X_copy['Vertical_Distance_To_Hydrology']**2) ** 0.5
        X_copy['Elev_m_VDH'] = X_copy['Elevation'] - \
            X_copy['Vertical_Distance_To_Hydrology']
        X_copy['Hydro_p_Fire'] = X_copy['Horizontal_Distance_To_Hydrology'] + \
            X_copy['Horizontal_Distance_To_Fire_Points']
        X_copy['Hydro_m_Fire'] = X_copy['Horizontal_Distance_To_Hydrology'] - \
            X_copy['Horizontal_Distance_To_Fire_Points']
        X_copy['Hydro_p_Road'] = X_copy['Horizontal_Distance_To_Hydrology'] + \
            X_copy['Horizontal_Distance_To_Roadways']
        X_copy['Hydro_m_Road'] = X_copy['Horizontal_Distance_To_Hydrology'] - \
            X_copy['Horizontal_Distance_To_Roadways']
        X_copy['Fire_p_Road'] = X_copy['Horizontal_Distance_To_Fire_Points'] + \
            X_copy['Horizontal_Distance_To_Roadways']
        X_copy['Fire_m_Road'] = X_copy['Horizontal_Distance_To_Fire_Points'] - \
            X_copy['Horizontal_Distance_To_Roadways']

        X_copy['Soil'] = 0
        for i in range(1, 41):
            X_copy['Soil'] += i * X_copy['Soil_Type'+str(i)]

        X_copy['Wilderness_Area'] = 0
        for i in range(1, 5):
            X_copy['Wilderness_Area'] += i * X_copy['Wilderness_Area'+str(i)]

        new_features = ['Elevation',
                        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                        'Horizontal_Distance_To_Fire_Points', 'Elev_m_VDH', 'Dist_to_Hydrolody', 'Hydro_p_Fire',
                        'Hydro_m_Fire', 'Hydro_p_Road', 'Hydro_m_Road', 'Fire_p_Road', 'Fire_m_Road', 'Wilderness_Area', 'Soil']

        return X_copy[new_features]


class Numerical_Standardizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_columns = []
        self.binary_columns = []
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Identify numerical columns excluding binary columns
        self.numeric_columns = X.select_dtypes(include=['number']).columns
        self.binary_columns = [
            col for col in self.numeric_columns if sorted(X[col].unique()) == [0, 1]]
        self.numeric_columns = [
            col for col in self.numeric_columns if col not in self.binary_columns]

        # Fit the scaler on non-binary numerical columns
        self.scaler.fit(X[self.numeric_columns])
        return self

    def transform(self, X):
        # Standardize non-binary numerical columns
        X_transformed = X.copy()
        X_transformed[self.numeric_columns] = self.scaler.transform(
            X[self.numeric_columns])
        return X_transformed

# Define the OverSample12 function


def OverSample12(train_df):
    """
    Function to oversample cover_types 1 and 2 in the dataset.

    Inputs:
    train_df: DataFrame to edit

    Outputs:
    X: Resampled Design Matrix
    y: Resampled target Variable    
    """
    # Separate features and target
    X = train_df.drop(columns=['Id', 'Cover_Type'])
    y = train_df['Cover_Type']

    # Combine X and y back for resampling
    data = pd.concat([X, y], axis=1)

    # Separate the dataset by 'Cover_Type'
    cover_type_1 = data[data['Cover_Type'] == 1]
    cover_type_2 = data[data['Cover_Type'] == 2]
    other_cover_types = data[~data['Cover_Type'].isin([1, 2])]

    # Oversample 'Cover_Type' 1 and 2
    cover_type_1_oversampled = resample(
        cover_type_1, replace=True, n_samples=25000, random_state=42)
    cover_type_2_oversampled = resample(
        cover_type_2, replace=True, n_samples=25000, random_state=42)

    # Combine the oversampled and non-oversampled parts
    balanced_data = pd.concat(
        [cover_type_1_oversampled, cover_type_2_oversampled, other_cover_types])

    # Split again into features and target
    X = balanced_data.drop('Cover_Type', axis=1)
    y = balanced_data['Cover_Type']
    return X, y


# Load the training data
train_csv = os.path.join("/kaggle/input/forest-dsb-2023-v2", "train.csv")
train_df = pd.read_csv(train_csv)

# Oversample and split the data
X, y = OverSample12(train_df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# Define and fit the pipeline
pipeline = Pipeline(steps=[
    ('FeatEng', FeatureEngineering()),
    ('standardizer', Numerical_Standardizer()),
    ('classifier', ExtraTreesClassifier(n_estimators=900, n_jobs=-1, verbose=0, max_depth=50,
                                        min_samples_split=10, min_samples_leaf=4, criterion='entropy',
                                        class_weight={1: 8, 2: 8, 3: 2, 4: 1, 5: 1, 6: 2, 7: 1}, warm_start=True))
])
pipeline.fit(X_train, y_train)

# Load the test data, predict, and save the results
test_csv = os.path.join("/kaggle/input/forest-dsb-2023-v2", "test-full.csv")
test_df = pd.read_csv(test_csv)
predictions = pipeline.predict(test_df)
result_df = pd.DataFrame({'Id': test_df['Id'], 'Cover_Type': predictions})
result_df.to_csv('/kaggle/working/submission.csv', index=False)
