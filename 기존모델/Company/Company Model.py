
# Environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
#from keras import backend as K
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import recall_score, precision_score, f1_score
import logging
tf.get_logger().setLevel(logging.ERROR)
import sys
np.set_printoptions(threshold=sys.maxsize)
#from tensorflow import keras

#%% Metrics Define
METRICS = tfa.metrics.F1Score(
      num_classes = 6,
      average = 'micro',
      name ='f1_score',
      threshold = 0.5
      )

#%%
## Data Read
df=pd.read_csv('Data/Company_Main_Data.csv')
df_sample = pd.read_csv('Data/Company_Val_Data.csv')

#%%
Scalar_Index =[
    'Sales',
    'Income',
    'Asset',
    'Capital'             
    ]

scaler = StandardScaler()
scaler.fit(df[Scalar_Index])
df[Scalar_Index] = scaler.transform(df[Scalar_Index])
#scaler.fit(df_sample[Scalar_Index])
df_sample[Scalar_Index] = scaler.transform(df_sample[Scalar_Index])

df["Employees"] = np.log1p(df["Employees"])
df_sample["Employees"] = np.log1p(df_sample["Employees"])

#%%
## Data pre-processing
NUMERIC_COLUMN = [
    "Year",
    "Log_RnD_Fund",
    "Log_Duration",
    "N_of_SCI",
    "N_of_Paper",
    "N_Patent_App",
    "N_Patent_Reg",
    "N_of_Korean_Patent",
    "STP_Code_1_Weight",
    "STP_Code_2_Weight",
    "Application_Area_1_Weight",
    "Application_Area_2_Weight",
    'Sales',
    'Income',
    'Asset',
    'Capital',             
    'Sales_Income_Ratio',
    'Asset_Income_Ratio', 
    'Sales_Operation_Ratio', 
    'Expense_Ratio',
    'Debt_Ratio',
    'Employees'
]

CATEGORICAL_COLUMN = [
    "IPO",
    "Comp_Type",
    "Listed_Market",
    "Administration",
    "External_Audit",
    "Survival",
    "Venture",
    "Innobiz",
    "Mainbiz",
    "Closed",
    "Ten_Industry_11",
    "Researcher",    
    "Multi_Year",
    "RnD_Org",
    "STP_Code_11",
    "STP_Code_21",
    "Application_Area_1",
    "Application_Area_2",
    "Green_Tech",
    "SixT_2",
    "Econ_Social",
    "National_Strategy_2",
    "RnD_Stage",
    "Cowork_Cor",
    "Cowork_Uni",
    "Cowork_Inst",
    "Cowork_Abroad",
    "Cowork_etc",
]

LABEL_COLUMN = [ 
    "Comm_Success",
    "Comm_Success_Code1_4",
    "Comm_Success_Code2_5",
    "Comm_Success_Code3_6",
]


#%%
## Define feature_layer
feature_columns = []
for header in NUMERIC_COLUMN:
    feature_columns.append(feature_column.numeric_column(header,dtype=tf.dtypes.float64))
for header in CATEGORICAL_COLUMN:
    vocab = feature_column.categorical_column_with_vocabulary_list(header, df[header].unique())
    feature_columns.append(feature_column.indicator_column(vocab))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#%%
## Data Split & shuffle
train_df, test_df = train_test_split(df,test_size=0.2)
train_df, val_df = train_test_split(train_df,test_size=0.25)

x_train = dict(train_df[NUMERIC_COLUMN + CATEGORICAL_COLUMN])
x_test = dict(test_df[NUMERIC_COLUMN + CATEGORICAL_COLUMN])
x_val = dict(val_df[NUMERIC_COLUMN + CATEGORICAL_COLUMN])
x_sample = dict(df_sample[NUMERIC_COLUMN + CATEGORICAL_COLUMN])

y_train = train_df[LABEL_COLUMN]
y_test = test_df[LABEL_COLUMN]
y_val = val_df[LABEL_COLUMN]
y_sample = df_sample[LABEL_COLUMN]

#%% Main Module

## EarlyStopping Condition
cb = tf.keras.callbacks.EarlyStopping( 
    monitor='loss', 
    min_delta = 0,
    patience=50, 
    verbose=1, 
    mode='min', 
    baseline=None, 
    restore_best_weights=True
    ) 

## Hyper-Parameter
Batch_Size = 256
Dim = 64
Dropout = 0.1
Alpha=[0.500, 0.965, 0.600, 0.930]
Epochs= 100

## Build model 
model = tf.keras.models.Sequential([
        feature_layer,
        tf.keras.layers.Dense(units=Dim, activation='relu'),
        tf.keras.layers.Dropout(rate=Dropout),
        tf.keras.layers.Dense(units=Dim, activation='relu'),
        tf.keras.layers.Dropout(rate=Dropout),
        tf.keras.layers.Dense(units=Dim, activation='relu'),
        tf.keras.layers.Dropout(rate=Dropout),        
        tf.keras.layers.Dense(units=y_train.shape[1], activation='sigmoid')
    ])
    
model.compile(optimizer='adam',
                  loss= tfa.losses.SigmoidFocalCrossEntropy(gamma = 2, alpha=Alpha),
                  metrics=METRICS
    )
            
model.fit(
        x_train, y_train,
        batch_size=Batch_Size,
        epochs=Epochs,
        verbose=2,
        validation_data=(x_val, y_val),
        callbacks=cb,
    )

model.save("Company_Main")







