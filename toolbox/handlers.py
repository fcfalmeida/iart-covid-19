import pandas as pd
import json
import pickle
from toolbox.encoder import Encoder
from toolbox.algorithms import Algorithms
import toolbox.utils as utils

def handle_train(args):
    le = Encoder()

    algorithm = Algorithms[args.algorithm].value

    df = pd.read_csv(args.training_file)

    # Fill Province/State NA values (temp)
    df['Province/State'] = df['Province/State'].fillna('NA_' + df['Country/Region'])

    df = le.encode_dataframe(df)

    if args.gs_params_file:
        with open(args.gs_params_file, 'r') as f:
            parameters = json.load(f)

        model = utils.train_with_grid_search(df, args.target_col, args.test_size, algorithm, parameters)
    else:
        model = utils.train_model(df, args.target_col, args.test_size, algorithm)

    if args.save_model_file:
        pickle.dump(model, open(args.save_model_file, 'wb'))

def handle_predict(args):
    le = Encoder()

    df = pd.read_csv(args.test_file)

    # Fill Province/State NA values (temp)
    df['Province/State'] = df['Province/State'].fillna('NA_' + df['Country/Region'])

    df = le.encode_dataframe(df)

    model = pickle.load( open( args.model_file, "rb"))

    predictions = model.predict(df)
    predictions = le.decode_column(args.target_col, predictions)
    df[args.target_col] = predictions

    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))