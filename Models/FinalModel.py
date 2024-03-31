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
        rs_df = X.copy()
        rs_df['Dist_to_Hydrolody'] = (
            rs_df['Horizontal_Distance_To_Hydrology']**2 + rs_df['Vertical_Distance_To_Hydrology']**2) ** 0.5
        rs_df['Elev_m_VDH'] = rs_df['Elevation'] - \
            rs_df['Vertical_Distance_To_Hydrology']
        rs_df['Elev_p_VDH'] = rs_df['Elevation'] + \
            rs_df['Vertical_Distance_To_Hydrology']
        rs_df['Elev_m_HDH'] = rs_df['Elevation'] - \
            rs_df['Horizontal_Distance_To_Hydrology']
        rs_df['Elev_p_HDH'] = rs_df['Elevation'] + \
            rs_df['Horizontal_Distance_To_Hydrology']
        rs_df['Elev_m_DH'] = rs_df['Elevation'] - rs_df['Dist_to_Hydrolody']
        rs_df['Elev_p_DH'] = rs_df['Elevation'] + rs_df['Dist_to_Hydrolody']
        rs_df['Hydro_p_Fire'] = rs_df['Horizontal_Distance_To_Hydrology'] + \
            rs_df['Horizontal_Distance_To_Fire_Points']
        rs_df['Hydro_m_Fire'] = rs_df['Horizontal_Distance_To_Hydrology'] - \
            rs_df['Horizontal_Distance_To_Fire_Points']
        rs_df['Hydro_p_Road'] = rs_df['Horizontal_Distance_To_Hydrology'] + \
            rs_df['Horizontal_Distance_To_Roadways']
        rs_df['Hydro_m_Road'] = rs_df['Horizontal_Distance_To_Hydrology'] - \
            rs_df['Horizontal_Distance_To_Roadways']
        rs_df['Fire_p_Road'] = rs_df['Horizontal_Distance_To_Fire_Points'] + \
            rs_df['Horizontal_Distance_To_Roadways']
        rs_df['Fire_m_Road'] = rs_df['Horizontal_Distance_To_Fire_Points'] - \
            rs_df['Horizontal_Distance_To_Roadways']
        rs_df['Elevation'] = rs_df['Elevation']**3

        rs_df['Soil'] = 0
        for i in range(1, 41):
            rs_df['Soil'] += i * rs_df['Soil_Type'+str(i)]

        rs_df['Wilderness_Area'] = 0
        for i in range(1, 5):
            rs_df['Wilderness_Area'] += i * rs_df['Wilderness_Area'+str(i)]

        new_features = ['Elevation',
                        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                        'Horizontal_Distance_To_Fire_Points', 'Elev_m_VDH', 'Dist_to_Hydrolody', 'Hydro_p_Fire',
                        'Hydro_m_Fire', 'Hydro_p_Road', 'Hydro_m_Road', 'Fire_p_Road', 'Fire_m_Road', 'Wilderness_Area', 'Soil']
        # 'Slope', 'Horizontal_Distance_To_Hydrology', 'Aspect', 'Vertical_Distance_To_Hydrology',

        return rs_df[new_features]


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


# Load the full test dataset
file_path = '../data/FULL-TEST.gz'
columns_to_use = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
    'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
    'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
    'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
    'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
    'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
    'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
    'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
    'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
    'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
    'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',
    'Cover_Type'
]


def OverSample12(train_df, i):
    """
    Function to oversample cover_types 1 and 2 in the dataset, if they are not empty.

    Inputs:
    train_df: DataFrame to edit

    Outputs:
    X: Resampled Design Matrix
    y: Resampled target Variable    
    """
    n_samples = 25000

    if i == 3:
        n_samples = 50000

    # Separate features and target
    X = train_df.drop(columns=['Id', 'Cover_Type'])
    y = train_df['Cover_Type']

    # Combine X and y back for resampling
    data = pd.concat([X, y], axis=1)

    # Separate the dataset by 'Cover_Type'
    cover_type_1 = data[data['Cover_Type'] == 1]
    cover_type_2 = data[data['Cover_Type'] == 2]
    other_cover_types = data[~data['Cover_Type'].isin([1, 2])]

    # Initialize an empty list to collect DataFrames for concatenation
    dataframes_to_combine = [other_cover_types]

    # Check if cover_type_1 is not empty, then oversample
    if not cover_type_1.empty:
        cover_type_1_oversampled = resample(
            cover_type_1, replace=True, n_samples=n_samples, random_state=42)
        dataframes_to_combine.append(cover_type_1_oversampled)

    # Check if cover_type_2 is not empty, then oversample
    if not cover_type_2.empty:
        cover_type_2_oversampled = resample(
            cover_type_2, replace=True, n_samples=n_samples, random_state=42)
        dataframes_to_combine.append(cover_type_2_oversampled)

    # Combine the oversampled and non-oversampled parts
    balanced_data = pd.concat(dataframes_to_combine)

    # Split again into features and target
    X = balanced_data.drop('Cover_Type', axis=1)
    y = balanced_data['Cover_Type']

    return X, y


pipelines = {}
predictions = {}


for i in range(1, 5):
    test_csv = os.path.join(
        "/kaggle/input/forest-dsb-2023-v2", "test-full.csv")
    full_test_df = pd.read_csv(test_csv)
    full_test_df['ind'] = full_test_df.index
    mask = full_test_df['Wilderness_Area' + str(i)] == True
    full_test_df = full_test_df[mask]
    original_indices = full_test_df['ind'].tolist()
    full_test_df.drop(columns='ind', inplace=True)

    # Load the training data
    train_csv = os.path.join("/kaggle/input/forest-dsb-2023-v2", "train.csv")
    train_df = pd.read_csv(train_csv)

    mask = train_df['Wilderness_Area' + str(i)] == True
    train_df = train_df[mask]

    X, y = OverSample12(train_df, i)

    test_size = 0.2
    if i == 3:
        test_size = 0.4
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=7)

    pipeline = Pipeline(steps=[
        ('FeatEng', FeatureEngineering()),
        ('standardizer', Numerical_Standardizer()),
        ('classifier', ExtraTreesClassifier(n_estimators=900, n_jobs=-1, verbose=0, max_depth=50,
                                            min_samples_split=10, min_samples_leaf=4, criterion='entropy',
                                            class_weight={1: 8, 2: 8, 3: 2, 4: 1, 5: 1, 6: 2, 7: 1}, warm_start=True))
    ])
    pipeline.fit(X_train, y_train)
    pipelines[f'area{i}'] = pipeline

    predictions[i] = {'predictions': pipeline.predict(
        full_test_df.drop('Cover_Type', axis=1)), 'indices': original_indices}


# List to store temporary DataFrames
temp_dfs = []

# Iterate through each key in the predictions dictionary
for area, data in predictions.items():
    # Extract ids (indices) and predictions for the current area
    ids = data['indices']
    cover_types = data['predictions']

    # Create a temporary DataFrame and append it to the list
    temp_df = pd.DataFrame({
        'Id': ids,
        'Cover_Type': cover_types
    })
    temp_dfs.append(temp_df)

# Concatenate all DataFrames in the list into a single DataFrame
consolidated_predictions_df = pd.concat(
    temp_dfs).sort_values('Id').reset_index(drop=True)

consolidated_predictions_df['Id'] += 1

# Export to CSV
consolidated_predictions_df.to_csv(
    '/kaggle/working/submission.csv', index=False)
