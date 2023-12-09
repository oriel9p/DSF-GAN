from dsfgan import DSFGAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score, mean_squared_error, r2_score
import joblib
from ctgan.synthesizers.ctgan import CTGAN
import torch

def _load_data(data):
    train_set = data.sample(frac=0.7, random_state=42)
    val_set = data.drop(train_set.index)
    return train_set, val_set

def _evaluate_synthetic_data(gan_model, n, val_set, feedback_type):
    """
    sample N_train from the trained generator, train classifier/regressor
    and evaluate the performance using the real validation set
    :param gan_model: DSFGAN model (Object)
    :param n: number of samples (int)
    :param val_set: (dataframe/np array)
    :return: model (Object), performance metric (string), value (float)
    """
    syn_data = gan_model.sample(n)
    # Evaluate model
    if feedback_type == "classification":
        # Train model
        model = LogisticRegression()
        model.fit(syn_data.iloc[:, :-1], syn_data.iloc[:, -1])
        precision = precision_score(val_set.iloc[:, -1], model.predict(val_set.iloc[:, :-1]))
        recall = recall_score(val_set.iloc[:, -1], model.predict(val_set.iloc[:, :-1]))
        print(f'precision: {precision}, recall: {recall}')
    else:
        # train model
        model = LinearRegression()
        model.fit(syn_data.iloc[:, :-1], syn_data.iloc[:, -1])
        rmse = np.sqrt(mean_squared_error(val_set.iloc[:, -1], model.predict(val_set.iloc[:, :-1])))
        rsqrt = r2_score(val_set.iloc[:, -1], model.predict(val_set.iloc[:, :-1]))
        print(f'RMSE: {rmse}, R2: {rsqrt}')
    return rmse, rsqrt

def _sample_from_trained(model_path, dataset):
    # Sample and save to csv
    data = pd.read_csv(f'datasets/clean/{dataset}.csv')
    n = int(data.shape[0] * 0.7)
    loaded_model = joblib.load(model_path)
    syn = loaded_model.sample(n)
    syn.to_csv(f'{dataset}_synthetic.csv')

if __name__ == '__main__':
    feedback_type = "classification"
    dataset_name = "adult_scaled"
    # feedback_type = "regression"
    # dataset_name = "house_price"
    data = pd.read_csv(f'datasets/clean/{dataset_name}.csv')
    data = data.drop('Unnamed: 0', axis=1)
    print(f'data raw shape: {data.shape}')
    # FEEDBACK
    train_set, val_set = _load_data(data)
    n_train = train_set.shape[0]
    # DSFGAN Object
    epochs = 100
    batch_size = 500
    dsfgan = DSFGAN(feedback_type, val_set, n_train, epochs=epochs, batch_size=batch_size)
    # Discrete cols
    # discrete_columns = ["floors", "waterfront","view","condition","city"]
    discrete_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    dsfgan.fit(train_set)
    joblib.dump(dsfgan, f'trained_models/{dataset_name}_e{epochs}_b{batch_size}_feedback.pkl')
    # Eval synthetic data (initial, more comprehensive in notebook)
    metrics = _evaluate_synthetic_data(dsfgan, n_train, val_set, feedback_type)
    # NO FEEDBACK
    print("CTGAN:")
    ctgan_m = CTGAN(epochs=epochs, batch_size=batch_size)
    ctgan_m.fit(train_set)
    joblib.dump(ctgan_m, f'trained_models/{dataset_name}_e{epochs}_b{batch_size}_nofeedback.pkl')
    # Eval synthetic data (initial, more comprehensive in notebook)
    metrics = _evaluate_synthetic_data(dsfgan, n_train, val_set, feedback_type)








