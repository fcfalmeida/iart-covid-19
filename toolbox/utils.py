from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from toolbox.algorithms import Algorithms
from toolbox.encoder import Encoder

# Train a model to predict a given dataframe column
def train_model(df, target_col, test_size, algorithm):
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    model = algorithm.fit(x_train, y_train)
    predictions = model.predict(x_test)
    
    print('R2 Score: ', r2_score(y_test, predictions))
    print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', mean_squared_error(y_test, predictions))

    return model

def train_with_grid_search(df, target_col, test_size, algorithm, parameters):
    clf = GridSearchCV(algorithm, parameters, verbose=2)
    model = train_model(df, target_col, test_size, clf)

    print(model.cv_results_)
    
    print('Best params:')
    print(model.best_params_)

    return model

def predict_values(model, test_df, target_col):
    le = Encoder()

    df = test_df.copy()
    df = le.encode_dataframe(df)

    predictions = model.predict(df)
    predictions = le.decode_column(target_col, predictions)
    df[target_col] = predictions

    df = le.decode_dataframe(df)

    return df

def alg_names():
    algs = map(lambda alg: alg.name, list(Algorithms))
    return list(algs)

def calc_metrics(y_test, predictions):
    print('R2 Score: ', r2_score(y_test, predictions))
    print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))
    print('Mean Square Error', mean_squared_error(y_test, predictions))
