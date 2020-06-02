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
import datetime
from scipy.stats import boxcox
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/Train.csv')
riders = pd.read_csv('data/riders.csv')
df = train.merge(riders, how='left', on='Rider Id')

#Categorical variables
df['User Id'] = pd.to_numeric(df['User Id'].str.split('User_Id_', n=1, expand = True)[1])
df = pd.get_dummies(df, columns=['Personal or Business'], drop_first=True)
df = pd.get_dummies(df, columns=['Platform Type'], drop_first=True)



##############################################################################
"""
Copyright (C) 2008 Leonard Norrgard <leonard.norrgard@gmail.com>
Copyright (C) 2015 Leonard Norrgard <leonard.norrgard@gmail.com>

This file is part of Geohash.

Geohash is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Geohash is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
License for more details.

You should have received a copy of the GNU Affero General Public
License along with Geohash.  If not, see
<http://www.gnu.org/licenses/>.
"""
from math import log10

#  Note: the alphabet in geohash differs from the common base32
#  alphabet described in IETF's RFC 4648
#  (http://tools.ietf.org/html/rfc4648)
__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__decodemap = { }
for i in range(len(__base32)):
    __decodemap[__base32[i]] = i
del i

def encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)
##############################################################################



#Transform latitude and longitude into geohashes
geo_df = df.loc[:, ['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long']]
geo_df['pickup'] = 0
geo_df['dest'] = 0
for i in range(len(geo_df)):
    geo_df.iloc[i, 4] = encode(geo_df.iloc[i, 0], geo_df.iloc[i, 1], precision=6)
    geo_df.iloc[i, 5] = encode(geo_df.iloc[i, 2], geo_df.iloc[i, 3], precision=6)

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
df['time_C-Pl'] = (df['Confirmation - Time'] - df['Placement - Time']).astype('timedelta64[s]').astype(np.int64)
df['time_AP-C'] = (df['Arrival at Pickup - Time'] - df['Confirmation - Time']).astype('timedelta64[s]').astype(np.int64)
df['time_P-AP'] = (df['Pickup - Time'] - df['Arrival at Pickup - Time']).astype('timedelta64[s]').astype(np.int64)

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
ls = ['pl', 'con', 'arr p', 'p']
ls_sin = ['pl_sin', 'con_sin', 'arr p_sin', 'p_sin']
ls_cos = ['pl_cos', 'con_cos', 'arr p_cos', 'p_cos']
for i in range(len(ls)):
    df[ls_sin[i]] = df[ls[i]].apply(lambda x: np.sin(x*(2.*np.pi/86400)))
    df[ls_cos[i]] = df[ls[i]].apply(lambda x: np.cos(x*(2.*np.pi/86400)))

#sin/cos transform 'Weekday'
df['weekday_sin'] = df['Pickup - Weekday (Mo = 1)'].apply(lambda x: np.sin(x*(2.*np.pi/7)))
df['weekday_cos'] = df['Pickup - Weekday (Mo = 1)'].apply(lambda x: np.cos(x*(2.*np.pi/7)))

#sin/cos transform 'Day of Month'
df['day_month_sin'] = df['Pickup - Day of Month']. apply(lambda x: np.sin(x*(2.*np.pi/31)))
df['day_month_cos'] = df['Pickup - Day of Month']. apply(lambda x: np.cos(x*(2.*np.pi/31)))

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
df['y_tf'] = boxcox(df['Time from Pickup to Arrival'])[0]

#Remove outliers using the IQR.
Q1 = df['y_tf'].quantile(0.25)
Q3 = df['y_tf'].quantile(0.75)
IQR = Q3 - Q1
df = df.drop(df[(df['y_tf'] < (Q1 - 1.5 * IQR)) | (df['y_tf'] > (Q3 + 1.5 * IQR))].index)

#Rank riders by weighted rating value and efficiency
total = sum(riders['No_of_Ratings'])
df['ranking'] = df['Average_Rating'] * df['No_of_Ratings'] / total
df['deliveries_per_day'] = df['No_Of_Orders'] / df['Age']

model_features = ['User Id', 'dest_geohash', 'pickup_geohash', 'time_C-Pl', 'time_AP-C', 'time_P-AP', 'Distance (KM)', 'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'pl', 'con', 'arr p', 'p',
                    'weekday_sin', 'weekday_cos', 'day_month_sin', 'day_month_cos', 'ranking', 'deliveries_per_day', 'pl_sin', 'con_sin', 'arr p_sin', 'p_sin', 'pl_cos', 'con_cos', 'arr p_cos', 'p_cos']

y_train = df[['Time from Pickup to Arrival']]
X_train = df[model_features]

# Fit model
regressor = LinearRegression(normalize=True)
print ("Training Model...")
regressor.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/regressor.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(regressor, open(save_path,'wb'))
