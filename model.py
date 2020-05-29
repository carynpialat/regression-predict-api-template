"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import datetime
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    df = feature_vector_df

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    train_data = pd.read_csv('utils/data/train_data.csv')
    riders = pd.read_csv('utils/data/riders.csv')
    print(df.columns)
    
    #Drop unnecessary columns
    df = df.drop([col for col in df.columns if 'Arrival at Destination' in col], axis=1)
    df = df.drop('Precipitation in millimeters', axis=1)
    df = df.drop('Vehicle Type', axis=1)
    ls = ['No_Of_Orders','Age','Average_Rating','No_of_Ratings']
    cols = list(df.columns)
    for i in range(len(cols)):
        if cols[i] in ls:
            df = df.drop(cols[i], axis=1)


    #Make 'User Id' column numeric. Doing this effectively replicates the result of performing label encoding on the column.
    df['User Id'] = pd.to_numeric(df['User Id'].str.split('User_Id_', n=1, expand = True)[1])

    #One hot encode 'Personal or Business' and 'Platform Type' columns. Drop first column of each attribute to avoid the dummy variable trap.
    df = pd.get_dummies(df, columns=['Personal or Business'], drop_first=True)
    df = pd.get_dummies(df, columns=['Platform Type'], drop_first=True)
    print(df.columns)
    
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


    #Transform training data latitude and longitude into geohashes
    geo_df = train_data.loc[:, ['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long']]
    geo_df['pickup'] = 0
    geo_df['dest'] = 0
    for i in range(len(geo_df)):
        geo_df.iloc[i, 4] = encode(geo_df.iloc[i, 0], geo_df.iloc[i, 1], precision=6)
        geo_df.iloc[i, 5] = encode(geo_df.iloc[i, 2], geo_df.iloc[i, 3], precision=6)

    # Make a dictionary of geohash labels
    labels = list(set(list(geo_df['pickup']) + list(geo_df['dest'])))
    vals = [i + 1 for i in list(range(0, len(labels)))]
    geohash_dict = dict(zip(labels, vals))

    #Transform test data latitude and longitude into geohashes
    geo_df = df.loc[:, ['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long']]
    geo_df['pickup'] = 0
    geo_df['dest'] = 0
    for i in range(len(geo_df)):
        geo_df.iloc[i, 4] = encode(geo_df.iloc[i, 0], geo_df.iloc[i, 1], precision=6)
        geo_df.iloc[i, 5] = encode(geo_df.iloc[i, 2], geo_df.iloc[i, 3], precision=6)

    #Transform geohash labels using the dictionary
    geo_df['pickup_label'] = geo_df['pickup'].apply(lambda i: geohash_dict[i] if i in geohash_dict.keys() else 0)
    geo_df['dest_label'] = geo_df['dest'].apply(lambda i: geohash_dict[i] if i in geohash_dict.keys() else 0)
    
    #Add to df
    df['pickup_geohash'] = geo_df['pickup_label']
    df['dest_geohash'] = geo_df['dest_label']
    print(df.columns)

    #Transform time columns into 24 hour format
    df['Placement - Time'] = pd.to_datetime(df['Placement - Time'], format='%I:%M:%S %p')
    df['Confirmation - Time'] = pd.to_datetime(df['Confirmation - Time'], format='%I:%M:%S %p')
    df['Arrival at Pickup - Time'] = pd.to_datetime(df['Arrival at Pickup - Time'], format='%I:%M:%S %p')
    df['Pickup - Time'] = pd.to_datetime(df['Pickup - Time'], format='%I:%M:%S %p')

    #Calculate intervals between all time columns. This format is the same as the given dependent variable, 'Time from Pickup to Arrival'
    df['time_Con - Pl'] = (df['Confirmation - Time'] - df['Placement - Time']).astype('timedelta64[s]').astype(np.int64)
    df['time_Arr P - Con'] = (df['Arrival at Pickup - Time'] - df['Confirmation - Time']).astype('timedelta64[s]').astype(np.int64)
    df['time_P - Arr P'] = (df['Pickup - Time'] - df['Arrival at Pickup - Time']).astype('timedelta64[s]').astype(np.int64)

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
    print(df.columns)

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
    print(df.columns)

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

    #Format dataset to have the correct columns
    df = df.drop(['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 'Pickup - Time','User Id', 'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long', 'Rider Id', 'No_Of_Orders','Age','Average_Rating', 'weighted_rating','No_of_Ratings'], axis=1)
    print(df.columns)
    print(type(df))
    print("Finished job")

    # ------------------------------------------------------------------------

    return df

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
