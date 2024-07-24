import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Import data and brief data exploration to understand what we are dealing with

path_dataset = "D:/centennial/centennial 2024 summer/comp247/Assignment/Group Project/Killed_and_Seriously_Injured.csv"
df = pd.read_csv(path_dataset)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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

# Transform category in numerical to make Correlation MatriX diagram
df_encoded = df.copy()
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        df_encoded[column] = pd.factorize(df_encoded[column])[0]

# Correlation Martrix
plt.figure(figsize=(20, 16))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

#only 77 fatal incidents reported on dataset
df['FATAL_NO'].value_counts()

#First check if there is any difference between x and y - long and lat

# Create scatter plot for x and y
plt.scatter(df['X'], df['Y'], color='blue', marker='o')
plt.title('Scatter Plot for x and y location')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Create scatter plot for LATITUDE and LONGITUDE
plt.scatter(df['LONGITUDE'], df['LATITUDE'], color='green', marker='o')
plt.title('Scatter Plot for Lat and Long location')
plt.xlabel('LATITUDE')
plt.ylabel('LONGITUDE')
plt.show()
# we realize there is no difference now check the number of fatal accident to see if there is any correlation between location and fatal accidents
subset = df[df['ACCLASS'] == 'Fatal']
plt.scatter(subset['X'], subset['Y'], color='red', label='fatality in area')
plt.title('Scatter Plot by Fatality')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(title='Fatality')
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
    counts = []
    for values in df_incidents[category]:
        count_yes = values.count('Yes')
        counts.append(count_yes)
    df_incidents[category + 'counter'] = counts

# Reset index
df_incidents.reset_index(inplace=True)


# df_incidents.to_csv('C:/Users/aegno/OneDrive - Centennial College/Classes/IV semester/Supervised Learning - comp247/project/incidents_df2.csv', index=False)

#statistical check
df_incidents.describe()

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
df_incidents['ACCLASS'] = df_incidents['ACCLASS'].replace(acclass_mapping)


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

# Create a new DataFrame with the transformed data and column names
X = pd.DataFrame(X_transformed, columns=all_column_names)

# Print class distribution of the target variable
print("Class distribution in the target variable:")
print(y.value_counts(normalize=True))

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"\n{model_name} Results:")
    print(f"Cross-Validation Mean Accuracy: {cv_scores.mean()}")
    
    # Training and evaluation on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

# Train and evaluate models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Support Vector Machine": SVC(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "Neural Network"]:
        train_and_evaluate(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    else:
        train_and_evaluate(model, X_train, X_test, y_train, y_test, name)



# #divide dataset in multiple relevent datasets
# driver_df = df[['ACCLASS', 'INVAGE', 'DRIVACT', 'DRIVCOND' ]]
# Ped_df = df[['ACCLASS', 'INVAGE', 'PEDTYPE', 'PEDACT', 'PEDCOND' ]]
# cycl_df = df[['ACCLASS', 'INVAGE', 'CYCLISTYPE', 'CYCACT', 'CYCCOND' ]]
# env_df = df[['ACCLASS', 'ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY', 'LIGHT','RDSFCOND']]


# #work on driver dataset

# #drop null values
# driver_df = driver_df.dropna(subset=['INVAGE', 'DRIVACT','DRIVCOND','ACCLASS'])

# # Identify categorical columns and do one hot encoding
# categorical_columns = driver_df.select_dtypes(include=['object', 'category']).columns
# df_dummies = pd.get_dummies(driver_df, columns=categorical_columns)
# driver_df = df_dummies
# driver_df = driver_df.astype(int)

# correlations = driver_df.corr()['ACCLASS_Fatal'][:-1]
# sorted_correlations = correlations.abs().sort_values(ascending=False)

# print("Point-Biserial Correlations with target:")
# print(sorted_correlations)

# #above correlation shows that age does not have significant impact so we can delete that from our data
# driver_df_final = driver_df[['ACCLASS_Fatal','DRIVACT_Exceeding Speed Limit', 'DRIVACT_Following too Close', 'DRIVACT_Lost control', 'DRIVCOND_Ability Impaired, Drugs', 'DRIVACT_Improper Turn', 'DRIVCOND_Inattentive', 'DRIVACT_Failed to Yield Right of Way', 'INVAGE_85 to 89', 'INVAGE_90 to 94', 'DRIVCOND_Had Been Drinking']]


# #work on ped dataset

# #drop null values
# Ped_df = Ped_df.dropna(subset = ['ACCLASS', 'INVAGE', 'PEDTYPE', 'PEDACT', 'PEDCOND' ])

# # Identify categorical columns and do one hot encoding
# categorical_columns = Ped_df.select_dtypes(include=['object', 'category']).columns
# df_dummies = pd.get_dummies(Ped_df, columns=categorical_columns)
# Ped_df = df_dummies
# Ped_df = Ped_df.astype(int)

# correlations = Ped_df.corr()['ACCLASS_Fatal'][:-1]
# sorted_correlations = correlations.abs().sort_values(ascending=False)

# print("Point-Biserial Correlations with target:")
# print(sorted_correlations)

# #above correlation shows that age does not have significant impact so we can delete that from our data
# Ped_df_final = Ped_df[['ACCLASS_Fatal','PEDACT_Crossing, no Traffic Control', 
#                        'PEDTYPE_Vehicle turns left while ped crosses with ROW at inter.',
#                        'PEDTYPE_Pedestrian hit at mid-block', 'PEDACT_Running onto Roadway',
#                        'PEDACT_On Sidewalk or Shoulder',
#                        'PEDTYPE_Pedestrian hit on sidewalk or shoulder',
#                        'PEDCOND_Had Been Drinking',
#                        'PEDTYPE_Pedestrian hit at parking lot',
#                        'PEDCOND_Medical or Physical Disability',
#                        'PEDTYPE_Vehicle turns right while ped crosses without ROW at inter.',
#                        'PEDACT_Crossing without right of way',
#                        'PEDACT_Crossing marked crosswalk without ROW',
#                        'PEDTYPE_Pedestrian involved in a collision with transit vehicle anywhere along roadway',
#                        'PEDCOND_Ability Impaired, Drugs']]

# print(Ped_df_final.describe())



# #work on cycl_df

# #drop null values
# cycl_df = cycl_df.dropna(subset = ['ACCLASS', 'INVAGE', 'CYCLISTYPE', 'CYCACT', 'CYCCOND' ])

# # Identify categorical columns and do one hot encoding
# categorical_columns = cycl_df.select_dtypes(include=['object', 'category']).columns
# df_dummies = pd.get_dummies(cycl_df, columns=categorical_columns)
# cycl_df = df_dummies
# cycl_df = cycl_df.astype(int)

# correlations = cycl_df.corr()['ACCLASS_Fatal'][:-1]
# sorted_correlations = correlations.abs().sort_values(ascending=False)

# #print("Point-Biserial Correlations with target:")
# #print(sorted_correlations)

# #above correlation shows that age does not have significant impact so we can delete that from our data
# cycl_df_final = cycl_df[['ACCLASS_Fatal','INVAGE_70 to 74', 
#                        'CYCLISTYPE_Cyclist struck at PXO(cyclist either travel in same dir. as veh. or ride across xwalk)',
#                        'CYCLISTYPE_Cyclist makes u-turn in-front of driver.', 'INVAGE_5 to 9',
#                        'CYCCOND_Inattentive',
#                        'CYCLISTYPE_Motorist without ROW drives into path of cyclist at inter, lnwy, dwy-Driver not turn.',
#                        'CYCLISTYPE_Cyclist strikes a parked vehicle.',
#                        'CYCLISTYPE_Motorist turned left across cyclists path.',
#                        'CYCACT_Lost control',
#                        'CYCLISTYPE_Motorist turning right on green or amber at signalized intersection strikes cyclist.',
#                        'CYCACT_Failed to Yield Right of Way',
#                        'CYCLISTYPE_Motorist turning right on red at signalized intersection strikes cyclist.',
#                        'CYCACT_Improper Turn',
#                        'CYCLISTYPE_Cyclist struck opened vehicle door',
#                        'CYCACT_Improper Lane Change']]

# print(cycl_df_final.describe())



# env_df = df[['ACCLASS', 'ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY', 'LIGHT','RDSFCOND']]

# # Identify categorical columns and do one hot encoding
# categorical_columns = env_df.select_dtypes(include=['object', 'category']).columns
# df_dummies = pd.get_dummies(env_df, columns=categorical_columns)
# env_df = df_dummies
# env_df = env_df.astype(int)

# correlations = env_df.corr()['ACCLASS_Fatal'][:-1]
# sorted_correlations = correlations.abs().sort_values(ascending=False)

# print("Point-Biserial Correlations with target:")
# print(sorted_correlations)

# #above correlation shows that age does not have significant impact so we can delete that from our data
# env_df_final = env_df[['TRAFFCTL_No Control','LIGHT_Dark', 'ROAD_CLASS_Laneway', 'VISIBILITY_Snow', 'RDSFCOND_Ice']]

# print(env_df_final.describe())


