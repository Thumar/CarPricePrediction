from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import tensorflow as tf

names = [
    'symboling',
    'normalized-losses',
    'make',
    'fuel-type',
    'aspiration',
    'num-of-doors',
    'body-style',
    'drive-wheels',
    'engine-location',
    'wheel-base',
    'length',
    'width',
    'height',
    'curb-weight',
    'engine-type',
    'num-of-cylinders',
    'engine-size',
    'fuel-system',
    'bore',
    'stroke',
    'compression-ratio',
    'horsepower',
    'peak-rpm',
    'city-mpg',
    'highway-mpg',
    'price',
]

# We also have to specify dtypes.
dtypes = {
    'symboling': np.int32,
    'normalized-losses': np.float32,
    'make': str,
    'fuel-type': str,
    'aspiration': str,
    'num-of-doors': str,
    'body-style': str,
    'drive-wheels': str,
    'engine-location': str,
    'wheel-base': np.float32,
    'length': np.float32,
    'width': np.float32,
    'height': np.float32,
    'curb-weight': np.float32,
    'engine-type': str,
    'num-of-cylinders': str,
    'engine-size': np.float32,
    'fuel-system': str,
    'bore': np.float32,
    'stroke': np.float32,
    'compression-ratio': np.float32,
    'horsepower': np.float32,
    'peak-rpm': np.float32,
    'city-mpg': np.float32,
    'highway-mpg': np.float32,
    'price': np.float32,
}


def raw_dataframe():
    df = pd.read_csv('imports-85.data', names=names, dtype=dtypes, na_values="?")
    return df


def load_data(y_name="price", train_fraction=0.7, seed=None):
    # Load the raw data columns.
    data = raw_dataframe()

    # Delete rows with unknowns
    data = data.dropna()

    # Shuffle the data
    np.random.seed(seed)

    # Split the data into train/test subsets.
    x_train = data.sample(frac=train_fraction, random_state=seed)
    x_test = data.drop(x_train.index)

    # Extract the label from the features DataFrame.
    y_train = x_train.pop(y_name)
    y_test = x_test.pop(y_name)

    return (x_train, y_train), (x_test, y_test)


def features_columns():
    make = tf.feature_column.categorical_column_with_hash_bucket('make', 50)

    fuel_type = tf.feature_column.categorical_column_with_vocabulary_list('fuel-type',
                                                                          vocabulary_list=['diesel', 'gas'])
    aspiration = tf.feature_column.categorical_column_with_vocabulary_list('aspiration',
                                                                           vocabulary_list=['std', 'turbo'])
    num_of_doors = tf.feature_column.categorical_column_with_vocabulary_list('num-of-doors',
                                                                             vocabulary_list=['two', 'four'])
    body_style = tf.feature_column.categorical_column_with_vocabulary_list('body-style',
                                                                           vocabulary_list=['hardtop', 'wagon', 'sedan',
                                                                                            'hatchback', 'convertible'])
    drive_whells = tf.feature_column.categorical_column_with_vocabulary_list('drive-wheels',
                                                                             vocabulary_list=['4wd', 'rwd', 'fwd'])
    engine_location = tf.feature_column.categorical_column_with_vocabulary_list('engine-location',
                                                                                vocabulary_list=['front', 'rear'])

    engine_type = tf.feature_column.categorical_column_with_vocabulary_list('engine-type',
                                                                            ['dohc', 'dohcv', 'l', 'ohc', 'ohcf',
                                                                             'ohcv',
                                                                             'rotor'])
    num_of_cylinders = tf.feature_column.categorical_column_with_vocabulary_list('num-of-cylinders',
                                                                                 ['eight', 'five', 'four', 'six',
                                                                                  'three',
                                                                                  'twelve', 'two'])
    fuel_system = tf.feature_column.categorical_column_with_vocabulary_list('fuel-system',
                                                                            ['1bbl', '2bbl', '4bbl', 'idi', 'mfi',
                                                                             'mpfi',
                                                                             'spdi', 'spfi'])

    feature_columns = [

        tf.feature_column.embedding_column(make, 3),
        tf.feature_column.indicator_column(fuel_type),
        tf.feature_column.indicator_column(aspiration),
        tf.feature_column.indicator_column(num_of_doors),
        tf.feature_column.indicator_column(body_style),
        tf.feature_column.indicator_column(drive_whells),
        tf.feature_column.indicator_column(engine_location),
        tf.feature_column.indicator_column(engine_type),

        tf.feature_column.indicator_column(num_of_cylinders),
        tf.feature_column.indicator_column(fuel_system),

        tf.feature_column.numeric_column('symboling'),
        tf.feature_column.numeric_column('normalized-losses'),
        tf.feature_column.numeric_column('wheel-base'),
        tf.feature_column.numeric_column('length'),
        tf.feature_column.numeric_column('width'),
        tf.feature_column.numeric_column('height'),
        tf.feature_column.numeric_column('curb-weight'),

        tf.feature_column.numeric_column('engine-size'),
        tf.feature_column.numeric_column('bore'),
        tf.feature_column.numeric_column('stroke'),
        tf.feature_column.numeric_column('compression-ratio'),
        tf.feature_column.numeric_column('horsepower'),
        tf.feature_column.numeric_column('peak-rpm'),
        tf.feature_column.numeric_column('city-mpg'),
        tf.feature_column.numeric_column('highway-mpg'), ]

    return feature_columns
