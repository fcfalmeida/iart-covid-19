import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
import json
import pickle
from toolbox.encoder import Encoder
from toolbox.algorithms import Algorithms
import toolbox.utils as utils

le = Encoder()

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train')
parser_train.add_argument('training_file', help='Path to the file that contains the training data')
parser_train.add_argument('algorithm', help='The regression algorithm to use', choices=utils.alg_names())
parser_train.add_argument('target_col', help='The target column the model should be able to predict')
parser_train.add_argument('test_size', type=float, help='Percent size of test data for train_test_split')
parser_train.add_argument('-gs', '--gsearch', dest='gs_params_file', 
    help='When specified, the model will be trained using GridSearch. \
        Takes as argument a JSON file containing parameters that should be used in GridSearch')
parser_train.add_argument('-s', '--save', dest="save_model_file", help="When specified, saves the trained model on target file. \
    Takes as argument the name of a file. If the file doesn't exist, it will be created")

args = parser.parse_args()

algorithm = Algorithms[args.algorithm].value

df = pd.read_csv(args.training_file)

# Fill Province/State NA values (temp)
df['Province/State'] = df['Province/State'].fillna('NA_' + df['Country/Region'])

df = le.encode_dataframe(df)

if (args.gs_params_file):
    with open(args.gs_params_file, 'r') as f:
        parameters = json.load(f)

    model = utils.train_with_grid_search(df, args.target_col, args.test_size, algorithm, parameters)
else:
    model = utils.train_model(df, args.target_col, args.test_size, algorithm)

if (args.save_model_file):
    pickle.dump(model, open(args.save_model_file, 'wb'))
