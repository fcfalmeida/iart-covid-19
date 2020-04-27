import argparse
import toolbox.utils as utils
import toolbox.handlers as handlers

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train')
parser_train.set_defaults(func=handlers.handle_train)
parser_train.add_argument('training_file', help='Path to the .csv file that contains the training data')
parser_train.add_argument('algorithm', help='The regression algorithm to use', choices=utils.alg_names())
parser_train.add_argument('target_col', help='The target column the model should be able to predict')
parser_train.add_argument('test_size', type=float, help='Percent size of test data for train_test_split')
parser_train.add_argument('-gs', '--gsearch', dest='gs_params_file', 
    help='When specified, the model will be trained using GridSearch. \
        Takes as argument a JSON file containing parameters that should be used in GridSearch')
parser_train.add_argument('-s', '--save', dest="save_model_file", help="When specified, saves the trained model on target file. \
    Takes as argument the name of a file. If the file doesn't exist, it will be created")

parser_predict = subparsers.add_parser('predict')
parser_predict.set_defaults(func=handlers.handle_predict)
parser_predict.add_argument('test_file', help='Path to the .csv test file. Should contain rows without the column that will be predicted')
parser_predict.add_argument('model_file', help='Path to a file which contains a serialized sklearn model')
parser_predict.add_argument('target_col', help='Name of the column to predict')
parser_predict.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the predictions on a .csv file. Takes as argument the name of the file')

args = parser.parse_args()
args.func(args)
