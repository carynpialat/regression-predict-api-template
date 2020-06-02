"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
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
    train = pd.read_csv('utils/data/Train.csv')
    riders = pd.read_csv('utils/data/Riders.csv')

    #Categorical variables
    df['User Id'] = pd.to_numeric(df['User Id'].str.split('User_Id_', n=1, expand = True)[1])
    df = pd.get_dummies(df, columns=['Personal or Business'], drop_first=True)
    df = pd.get_dummies(df, columns=['Platform Type'], drop_first=True)
    print(df.columns)


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



    #Transform training data latitude and longitude into geohashes
    geo_df = train.loc[:, ['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long']]
    geo_df['pickup'] = 0
    geo_df['dest'] = 0
    for i in range(len(geo_df)):
        geo_df.iloc[i, 4] = encode(geo_df.iloc[i, 0], geo_df.iloc[i, 1], precision=6)
        geo_df.iloc[i, 5] = encode(geo_df.iloc[i, 2], geo_df.iloc[i, 3], precision=6)

    # Make a dictionary of geohash labels
    labels = list(set(list(geo_df['pickup']) + list(geo_df['dest'])))
    vals = [i + 1 for i in list(range(0, len(labels)))]
    geohash_dict = dict(zip(labels, vals))

    #Transform testing data latitude and longitude into geohashes
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

    #Rank riders by weighted rating value and efficiency
    total = sum(riders['No_of_Ratings'])
    df['ranking'] = df['Average_Rating'] * df['No_of_Ratings'] / total
    df['deliveries_per_day'] = df['No_Of_Orders'] / df['Age']

    model_features = ['User Id', 'dest_geohash', 'pickup_geohash', 'time_C-Pl', 'time_AP-C', 'time_P-AP', 'Distance (KM)', 'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'pl', 'con', 'arr p', 'p',
                        'weekday_sin', 'weekday_cos', 'day_month_sin', 'day_month_cos', 'ranking', 'deliveries_per_day', 'pl_sin', 'con_sin', 'arr p_sin', 'p_sin', 'pl_cos', 'con_cos', 'arr p_cos', 'p_cos']

    df = df[model_features]

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
