import argparse
import toolbox.utils as utils
import toolbox.handlers as handlers

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser('train', help='Train a model using one of the available algorithms')
parser_train.set_defaults(func=handlers.handle_train)
parser_train.add_argument('training_file', help='Path to the .csv file that contains the training data. \
    Must not contain missing values. The prep fillna command can be used to replace them.')
parser_train.add_argument('algorithm', help='The regression algorithm to use', choices=utils.alg_names())
parser_train.add_argument('target_col', help='The target column the model should be able to predict')
parser_train.add_argument('test_size', type=float, help='Percent size of test data for train_test_split')
parser_train.add_argument('-gs', '--gsearch', dest='gs_params_file', 
    help='When specified, the model will be trained using GridSearch. \
        Takes as argument a JSON file containing parameters that should be used in GridSearch')
parser_train.add_argument('-s', '--save', dest="save_model_file", help="When specified, saves the trained model on target file. \
    Takes as argument the name of a file. If the file doesn't exist, it will be created")

parser_predict = subparsers.add_parser('predict', help='Predict values using a previously trained model')
parser_predict.set_defaults(func=handlers.handle_predict)
parser_predict.add_argument('test_file', help='Path to the .csv test file. Should contain rows without the column that will be predicted \
    Must not contain missing values. The prep fillna command can be used to replace them.')
parser_predict.add_argument('model_file', help='Path to a file which contains a serialized sklearn model')
parser_predict.add_argument('target_col', help='Name of the column to predict')
parser_predict.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the predictions on a .csv file. Takes as argument the name of the file')

parser_prep = subparsers.add_parser('prep', help='Dataframe preprocessing utilities')
prep_subparsers = parser_prep.add_subparsers(help='prep help')

parser_prep_fillna = prep_subparsers.add_parser('fillna', help='Replace NA values with a given expression')
parser_prep_fillna.set_defaults(func=handlers.handle_prep_fillna)
parser_prep_fillna.add_argument('file', help='Path to the .csv file on which to fill the NA values')
parser_prep_fillna.add_argument('column', help='Name of the column for which NA values should be filled')
parser_prep_fillna.add_argument('expression', help='Valid Python expression by which NA values should be replaced. \
    If it contains spaces, it must be surrounded by double quotes. \
    Use df to refer to the dataframe, for example df["colname"]')
parser_prep_fillna.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the transformed dataframe on a .csv file. Takes as argument the name of the file')

parser_prep_drop = prep_subparsers.add_parser('drop', help='Drop a dataframe column')
parser_prep_drop.set_defaults(func=handlers.handle_prep_drop)
parser_prep_drop.add_argument('file', help='Path to the .csv file from which the column will be dropped')
parser_prep_drop.add_argument('column', help='Name of the column to drop')
parser_prep_drop.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the transformed dataframe on a .csv file. Takes as argument the name of the file')

parser_prep_transformdate = prep_subparsers.add_parser('transformdate', help='Creates a days time delta column from a date column')
parser_prep_transformdate.set_defaults(func=handlers.handle_prep_transformdate)
parser_prep_transformdate.add_argument('file', help='Path to the .csv file to transform')
parser_prep_transformdate.add_argument('date_col', help='Name of the date column')
parser_prep_transformdate.add_argument('new_col', help='Name of the new column in which the day difference will be stored')
parser_prep_transformdate.add_argument('refdate', help='Reference date from which the day difference will be calculated')
parser_prep_transformdate.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the transformed dataframe on a .csv file. Takes as argument the name of the file')

parser_prep_renamecol = prep_subparsers.add_parser('renamecol', help='Rename a column')
parser_prep_renamecol.set_defaults(func=handlers.handle_prep_renamecol)
parser_prep_renamecol.add_argument('file', help='Path to the .csv file to transform')
parser_prep_renamecol.add_argument('col', help='Name of the column to rename')
parser_prep_renamecol.add_argument('new_col', help='New name of the column')
parser_prep_renamecol.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the transformed dataframe on a .csv file. Takes as argument the name of the file')

parser_prep_custom = prep_subparsers.add_parser('custom', help='Custom set of transformations: replaces missing values on \
    the State/Province column, runs transformdate command on Date column, adds Population data.')
parser_prep_custom.set_defaults(func=handlers.handle_prep_custom)
parser_prep_custom.add_argument('file', help='Path to the .csv file to transform')
parser_prep_custom.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the transformed dataframe on a .csv file. Takes as argument the name of the file')

parser_prep_date_between = prep_subparsers.add_parser('datebetween', help='Transforms a dataset to only include rows with date\
    between two given dates. Includes both the first and second dates')
parser_prep_date_between.set_defaults(func=handlers.handle_prep_date_between)
parser_prep_date_between.add_argument('file', help='Path to the .csv file to transform')
parser_prep_date_between.add_argument('first_date', help='The first date in YYYY-mm-dd format')
parser_prep_date_between.add_argument('second_date', help='The second date in YYYY-mm-dd format')
parser_prep_date_between.add_argument('-o', '--outfile', dest='out_file',
    help='When specified, saves the transformed dataframe on a .csv file. Takes as argument the name of the file')

args = parser.parse_args()
args.func(args)
