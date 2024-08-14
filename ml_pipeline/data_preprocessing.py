import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils.logger import setup_logger
import pickle
import os

logger = setup_logger()
pd.set_option('future.no_silent_downcasting', True)

def load_and_preprocess_data(file_path, logger):

    logger.info(f"Loading data from {file_path}")
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Check if file exists and read the first few lines
        with open(file_path, 'r') as f:
            first_lines = [next(f) for _ in range(5)]
        logger.info(f"First few lines of the file:\n{''.join(first_lines)}")

        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError(f"The file {file_path} is empty.")
        
        # Import data
        logger.info("Reading CSV file...")
        df = pd.read_csv(file_path)
        logger.info(f"CSV file read successfully. Shape: {df.shape}")

        # Import data and brief data exploration to understand what we are dealing with
        df = pd.read_csv(file_path)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    except pd.errors.EmptyDataError:
        logger.error(f"The file {file_path} is empty or contains no data.")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the CSV file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {str(e)}")
        raise

    '''
    data exploration
    '''
    #print unique for each column
    for column in df.columns:
        print(f"{column}\n {df[column].unique()}")

    #count missing values for each column
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    #statistical check
    df.describe()

    '''
    Correlation Matrix (before any transformations)
    '''
    # Transform category in numerical to make Correlation MatriX diagram
    df_encoded = df.copy()
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            df_encoded[column] = pd.factorize(df_encoded[column])[0]

    # # Correlation Martrix
    # plt.figure(figsize=(20, 16))
    # sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    # plt.title('Correlation Matrix of Features')
    # plt.show()

    # plt.figure(figsize=(12, 10))
    # sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f', 
    #         mask=np.triu(np.ones_like(df.corr(), dtype=bool)),
    #         cbar_kws={'label': 'Correlation Coefficient'})
    # plt.title('Correlation Matrix of Features')
    # plt.tight_layout()
    # plt.show()

    #only 77 fatal incidents reported on dataset
    df['FATAL_NO'].value_counts()

    #First check if there is any difference between x and y - long and lat

    # # Create scatter plot for x and y
    # plt.scatter(df['X'], df['Y'], color='blue', marker='o')
    # plt.title('Scatter Plot for x and y location')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.show()

    # # Create scatter plot for LATITUDE and LONGITUDE
    # plt.scatter(df['LONGITUDE'], df['LATITUDE'], color='green', marker='o')
    # plt.title('Scatter Plot for Lat and Long location')
    # plt.xlabel('LATITUDE')
    # plt.ylabel('LONGITUDE')
    # plt.show()
    # we realize there is no difference now check the number of fatal accident to see if there is any correlation between location and fatal accidents
    # subset = df[df['ACCLASS'] == 'Fatal']
    # plt.scatter(subset['X'], subset['Y'], color='red', label='fatality in area')
    # plt.title('Scatter Plot by Fatality')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(title='Fatality')
    #there is no strong correlation and as we have many features we can ignore everything related to location


    df_incidents = df.groupby(['ACCNUM', 'DATE']).agg(list)


    #print unique for each column
    for column in df_incidents.columns:
        print(f"{column}\n {df[column].unique()}")

    #count missing values for each column
    print(df_incidents.info())
    print("\nMissing values:")
    print(df_incidents.isnull().sum())

    #handle bynary class
    categories = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH',
              'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']

    # Count 'Yes' for each category
    for category in categories:
        counts = [values.count('Yes') for values in df_incidents[category]]
        df_incidents[category + 'counter'] = counts

    # df_incidents.to_csv('C:/Users/aegno/OneDrive - Centennial College/Classes/IV semester/Supervised Learning - comp247/project/incidents_df2.csv', index=False)

    #statistical check
    df_incidents.describe()

    # Reset index
    df_incidents.reset_index(inplace=True)

    #feature selection
    features = [
        'ROAD_CLASS', 'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE',
        'PEDESTRIANcounter', 'CYCLISTcounter', 'AUTOMOBILEcounter', 'MOTORCYCLEcounter', 'TRUCKcounter', 'TRSN_CITY_VEHcounter', 'EMERG_VEHcounter', 'PASSENGERcounter', 
        'SPEEDINGcounter', 'AG_DRIVcounter', 'REDLIGHTcounter', 'ALCOHOLcounter', 'DISABILITYcounter'
    ]

    df_incidents = df_incidents[features]

    #delete lists with duplicate value in the dataset
    for col in features:
        if col in df_incidents.columns:
            first_values = []
            for value in df_incidents[col]:
                if isinstance(value, list):
                    first_values.append(value[0] if value else None)
                else:
                    first_values.append(value)
            df_incidents[col] = first_values



    #create Target y
    df_incidents['ACCLASS'].unique()
    df_incidents = df_incidents.dropna(subset=['ACCLASS']) # delete any row where the TARGET variable is null
    acclass_mapping = {'Fatal': 1, 'Non-Fatal Injury': 0, 'Property Damage O': 0}
    df_incidents['ACCLASS'] = df_incidents['ACCLASS'].replace(acclass_mapping).astype(int)


    df_majority = df_incidents[df_incidents['ACCLASS'] == 0]
    df_minority = df_incidents[df_incidents['ACCLASS'] == 1]
 
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results
 
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
    # Display new class counts
    df_upsampled.ACCLASS.value_counts()


    y = df_upsampled['ACCLASS']

    '''
    Final Data Preparation
    '''
    #create features X
    X = df_upsampled[features]
    X = X.drop('ACCLASS', axis=1)

    X.shape
    y.shape

    # List of categorical columns to transform
    categorical_features = ['ROAD_CLASS', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'DISTRICT']

    ordinal_encoder = OrdinalEncoder()
    knn_imputer = KNNImputer(n_neighbors=20)

    # Define the pipeline for categorical columns
    categorical_pipeline = Pipeline([
        ('ordinal_encoder', ordinal_encoder),
        ('knn_imputer', knn_imputer)
    ])

    # Create the ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    # Apply the pipeline to the DataFrame
    X_transformed = transformer.fit_transform(X)

    # Get the column names after transformation
    categorical_feature_names = transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)
    numerical_feature_names = X.columns.drop(categorical_features).tolist()

    all_column_names = categorical_feature_names.tolist() + numerical_feature_names

    if len(all_column_names) != X_transformed.shape[1]:
        raise ValueError("Mismatch between number of columns in transformed data and feature names.")
    
    # Create a new DataFrame with the transformed data and column names
    X = pd.DataFrame(X_transformed, columns=all_column_names)

    # After preprocessing and before splitting the data
    feature_names = X.columns.tolist()

    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
        
    # Verify that feature names were saved correctly
    try:
        with open('models/feature_names.pkl', 'rb') as f:
            loaded_feature_names = pickle.load(f)
        print("Loaded feature names:", loaded_feature_names)
    except Exception as e:
        print(f"Error loading feature names: {e}")
    
    '''
    Splitting Data
    '''
    # Print class distribution of the target variable
    print("Class distribution in the target variable:")
    print(y.value_counts(normalize=True))

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    '''
    Scaling Features
    '''
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Data preprocessing completed")
    #return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns
