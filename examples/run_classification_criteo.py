import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM

def print_feature_info(feature_columns):
    for fc in feature_columns:
        print(f"Feature: {fc.name}, Type: {type(fc)}, Dtype: {fc.dtype}, Vocab Size: {fc.vocabulary_size if hasattr(fc, 'vocabulary_size') else 'N/A'}")

# Load data
data = pd.read_csv('./criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
target = ['label']

# Fill na
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

# Feature hashing for sparse features
n_features = 1000  # You can adjust this value
hashers = {}
for feat in sparse_features:
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    # Convert the series to a list of single-element lists
    feature_list = [[str(x)] for x in data[feat]]
    hashed_feature = hasher.fit_transform(feature_list).toarray()
    for i in range(n_features):
        data[f"{feat}_hash_{i}"] = hashed_feature[:, i].copy()
    hashers[feat] = hasher
    # Debug"
    # print(f"Feature {feat} hashed into {n_features} features")

# Convert dense features to float32
for feat in dense_features:
    data[feat] = data[feat].astype('float32')

# Normalize dense features
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# Generate train and test data
train, test = train_test_split(data, test_size=0.2, random_state=2020)

# Prepare feature columns
fixlen_feature_columns = []
for feat in sparse_features:
    for i in range(n_features):
        fixlen_feature_columns.append(SparseFeat(f"{feat}_hash_{i}", vocabulary_size=2, embedding_dim=4, dtype=tf.int32))
fixlen_feature_columns += [DenseFeat(feat, 1, dtype=tf.float32) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# Debug: Print feature information
# print("\nLinear Feature Columns:")
# print_feature_info(linear_feature_columns)
# print("\nDNN Feature Columns:")
# print_feature_info(dnn_feature_columns)

train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

# Define the model
def create_and_fit_model(linear_feature_columns, dnn_feature_columns, train_model_input, train_target, **kwargs):
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['AUC'])
    history = model.fit(train_model_input, train_target, **kwargs)
    return model, history

# Create and train the model
model, history = create_and_fit_model(linear_feature_columns, dnn_feature_columns, 
                                      train_model_input, train[target].values,
                                      batch_size=256, epochs=10, verbose=2, validation_split=0.2)

# Evaluate the model
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))