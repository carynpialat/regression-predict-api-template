"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
pip install pygeohash
import pygeohash as pgh

# Fetch training data and preprocess for modeling
df = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')

#Delete attributes that are absent from test data except the dependent variable
df = df.drop([col for col in df.columns if 'Arrival at Destination' in col], axis=1)

#Drop columns that have more than 50% missing data
col_names = list(df.columns)
for i in range(len(col_names)):
    if (sum(df[col_names[i]].isna()) / len(df) * 100) > 50:
        df = df.drop(col_names[i], axis=1)

#Drop 'Vehicle Type' column because it is given as only one type
df = df.drop('Vehicle Type', axis=1)

#Make 'User Id' column numeric. Doing this effectively replicates the result of performing label encoding on the column.
df['User Id'] = pd.to_numeric(df['User Id'].str.split('User_Id_', n=1, expand = True)[1])

#One hot encode 'Personal or Business' and 'Platform Type' columns. Drop first column of each attribute to avoid the dummy variable trap.
df = pd.get_dummies(df, columns=['Personal or Business'], drop_first=True)
df = pd.get_dummies(df, columns=['Platform Type'], drop_first=True)

#Transform latitude and longitude into geohashes
geo_df = df.loc[:, ['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long']]
geo_df['pickup'] = 0
geo_df['dest'] = 0
for i in range(len(geo_df)):
    geo_df.iloc[i, 4] = pgh.encode(geo_df.iloc[i, 0], geo_df.iloc[i, 1], precision=6)
    geo_df.iloc[i, 5] = pgh.encode(geo_df.iloc[i, 2], geo_df.iloc[i, 3], precision=6)

# Make a dictionary of geohash labels
labels = list(set(list(geo_df['pickup']) + list(geo_df['dest'])))
vals = [i + 1 for i in list(range(0, len(labels)))]
geohash_dict = dict(zip(labels, vals))

#Transform geohash labels using the dictionary
geo_df['pickup_label'] = geo_df['pickup'].apply(lambda i: geohash_dict[i] if i in geohash_dict.keys() else 0)
geo_df['dest_label'] = geo_df['dest'].apply(lambda i: geohash_dict[i] if i in geohash_dict.keys() else 0)

#Add to df
df['pickup_geohash'] = geo_df['pickup_label']
df['dest_geohash'] = geo_df['dest_label']

#Transform time columns into 24 hour format
df['Placement - Time'] = pd.to_datetime(df['Placement - Time'], format='%I:%M:%S %p')
df['Confirmation - Time'] = pd.to_datetime(df['Confirmation - Time'], format='%I:%M:%S %p')
df['Arrival at Pickup - Time'] = pd.to_datetime(df['Arrival at Pickup - Time'], format='%I:%M:%S %p')
df['Pickup - Time'] = pd.to_datetime(df['Pickup - Time'], format='%I:%M:%S %p')

#Calculate intervals between all time columns. This format is the same as the given dependent variable, 'Time from Pickup to Arrival'
df['time_Con - Pl'] = (df['Confirmation - Time'] - df['Placement - Time']).astype('timedelta64[s]').astype(np.int64)
df['time_Arr P - Con'] = (df['Arrival at Pickup - Time'] - df['Confirmation - Time']).astype('timedelta64[s]').astype(np.int64)
df['time_P - Arr P'] = (df['Pickup - Time'] - df['Arrival at Pickup - Time']).astype('timedelta64[s]').astype(np.int64)

#Drop rows that has negative time intervals (eg. not possible for the confirmation to happen before the order is placed)
ls = [col for col in df if col.startswith('time')]
for i in range(len(ls)):
    df = df.drop(df[df[ls[i]] <= 0].index)

#Calculate time in seconds from midnight

df['pl'] = df['Placement - Time']. apply(lambda x: (x - pd.to_datetime('12:00:00 AM', format='%I:%M:%S %p')).total_seconds())
df['con'] = df['Confirmation - Time']. apply(lambda x: (x - pd.to_datetime('12:00:00 AM', format='%I:%M:%S %p')).total_seconds())
df['arr p'] = df['Arrival at Pickup - Time']. apply(lambda x: (x - pd.to_datetime('12:00:00 AM', format='%I:%M:%S %p')).total_seconds())
df['p'] = df['Pickup - Time']. apply(lambda x: (x - pd.to_datetime('12:00:00 AM', format='%I:%M:%S %p')).total_seconds())

#sin/cos transformation of time values

df['pl_sin'] = df['pl'].apply(lambda x: np.sin(x*(2.*np.pi/86400))) #86400 sec/d
df['pl_cos'] = df['pl'].apply(lambda x: np.cos(x*(2.*np.pi/86400)))

df['con_sin'] = df['con'].apply(lambda x: np.sin(x*(2.*np.pi/86400)))
df['con_cos'] = df['con'].apply(lambda x: np.cos(x*(2.*np.pi/86400)))

df['arr p_sin'] = df['arr p'].apply(lambda x: np.sin(x*(2.*np.pi/86400)))
df['arr p_cos'] = df['arr p'].apply(lambda x: np.cos(x*(2.*np.pi/86400)))

df['p_sin'] = df['p'].apply(lambda x: np.sin(x*(2.*np.pi/86400)))
df['p_cos'] = df['p'].apply(lambda x: np.cos(x*(2.*np.pi/86400)))

#Drop all but one columns that show multicolinearity
df['weekday'] = df['Pickup - Weekday (Mo = 1)']
df['month_day'] = df['Pickup - Day of Month']
ls = [col for col in df.columns if 'Weekday' in col] + [col for col in df.columns if 'Month' in col]
for i in range(len(ls)):
    df = df.drop(ls[i], axis=1)

#sin/cos transform 'Weekday'
df['weekday_sin'] = df['weekday'].apply(lambda x: np.sin(x*(2.*np.pi/7)))
df['weekday_cos'] = df['weekday'].apply(lambda x: np.cos(x*(2.*np.pi/7)))

#sin/cos transform 'Day of Month'
df['day_month_sin'] = df['month_day']. apply(lambda x: np.sin(x*(2.*np.pi/31)))
df['day_month_cos'] = df['month_day']. apply(lambda x: np.cos(x*(2.*np.pi/31)))

#Evaluate shortest times for target value
speed = df.loc[:, ['Time from Pickup to Arrival', 'Distance (KM)']]
speed['speed (km/h)'] = 0
for i in range(len(speed)):
    speed.iloc[i, 2] = speed.iloc[i, 1] / (speed.iloc[i, 0] / 3600)

df['speed (km/h)'] = speed['speed (km/h)']

#Drop rows that have speeds in excess of 110 km/h (max legal driving speed between Uganda and Kenya)
df = df.drop(df[df['speed (km/h)'] > 110].index)
df = df.drop('speed (km/h)', axis=1)

#Data shows many outliers to the right. Use boxcox transformation to adjust the y variable to a more normal distribution
from scipy.stats import boxcox
df['y_tf'] = boxcox(df['Time from Pickup to Arrival'])[0]

#Remove outliers using the IQR.
Q1 = df['y_tf'].quantile(0.25)
Q3 = df['y_tf'].quantile(0.75)
IQR = Q3 - Q1
df = df.drop(df[(df['y_tf'] < (Q1 - 1.5 * IQR)) | (df['y_tf'] > (Q3 + 1.5 * IQR))].index)

#Rank riders by weighted rating value and efficiency
riders['weighted_rating'] = 0
riders['deliveries_per_day'] = 0
total = sum(riders['No_of_Ratings'])
for i in range(len(riders)):
    riders.iloc[i, 5] = riders.iloc[i, 3] * (riders.iloc[i, 4] / total)
    riders.iloc[i, 6] = riders.iloc[i, 1] / riders.iloc[i, 2]
    
riders = riders.sort_values('weighted_rating', ascending=False).reset_index()
riders['ranking'] = riders.index

df = pd.merge(df, riders, how='left', left_on=['Rider Id'], right_on=['Rider Id'])

#Calculate mean temperature per hour
temp_adj = df.loc[:, ['Temperature', 'Placement - Time']]
temp_adj['hour'] = temp_adj['Placement - Time'].apply(lambda x: x.hour)
mean_temps = temp_adj.drop(temp_adj[temp_adj['Temperature'].isna()].index)
mean_temps = mean_temps.groupby(['hour'], as_index=False).mean()

#Replace nan Temperatures with mean per hour
a = temp_adj['Temperature'].isna()

for i in range(len(a)):
    if a.iloc[i] == True:
        temp_adj.iloc[i, 0] = mean_temps.loc[mean_temps['hour'] == temp_adj.iloc[i, 2], 'Temperature'].values[0]

df['temp_adj'] = temp_adj['Temperature']
df = df.drop('Temperature', axis=1)

df = df.set_index('Order No')

#Move dependent variable to the end
df['Time from Pickup to Arrival'] = df.pop('Time from Pickup to Arrival')

df = df.drop(['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 'Pickup - Time','User Id', 'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long', 'Rider Id', 'No_Of_Orders','Age','Average_Rating', 'weighted_rating','No_of_Ratings','y_tf'], axis=1)

#Create the matrix of features.
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values

#Training the Random Forest Regression model on the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
print ("Training Model...")
model = regressor.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../trained-models/model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(model, open(save_path,'wb'))
