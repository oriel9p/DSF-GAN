from dsfgan import DSFGAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score
# from sklearn.externals import joblib



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
    return precision, recall

# def _verbose_evaluation(model, dataset, epochs, batch_size, performance_metric, performance_value):


if __name__ == '__main__':
    feedback_type = "classification"
    dataset_name = "adult_scaled"
    data = pd.read_csv(f'datasets/clean/{dataset_name}.csv')
    data = data.drop('Unnamed: 0', axis=1)
    print(f'data raw shape: {data.shape}')
    train_set, val_set = _load_data(data)
    # DSFGAN Object
    n_train = train_set.shape[0]
    dsfgan = DSFGAN(feedback_type, val_set, n_train, epochs=100)
    # Discrete cols
    # discrete_columns = ["Gender","Email Opened","Email Clicked","Product page visit","Discount offered","Purchased"]
    discrete_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    dsfgan.fit(train_set)
    # Eval dataset
    print(_evaluate_synthetic_data(dsfgan, n_train, val_set, feedback_type))









