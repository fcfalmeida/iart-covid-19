import pandas as pd
import json
import pickle
from datetime import datetime
from toolbox.encoder import Encoder
from toolbox.algorithms import Algorithms
import toolbox.utils as utils

def handle_train(args):
    le = Encoder()

    algorithm = Algorithms[args.algorithm].value

    df = pd.read_csv(args.training_file)
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
    df = pd.read_csv(args.test_file)

    model = pickle.load(open(args.model_file, 'rb'))

    predictions = utils.predict_values(model, df, args.target_col)
    print(predictions)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))

def handle_prep_fillna(args):
    df = pd.read_csv(args.file)

    df[args.column] = df[args.column].fillna(eval(args.expression))
    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))

def handle_prep_drop(args):
    df = pd.read_csv(args.file)

    df = df.drop(args.column, axis=1)
    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))

def handle_prep_transformdate(args):
    df = pd.read_csv(args.file)
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    basedate = pd.Timestamp(args.refdate)

    df[args.new_col] = (df[args.date_col] - basedate).dt.days
    #df[args.date_col] = df[args.new_col]
    #df = df.drop(args.new_col, axis=1)
    #df = df.rename(columns={args.date_col: args.new_col})
    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))

def handle_prep_renamecol(args):
    df = pd.read_csv(args.file)
    df = df.rename(columns={args.col: args.new_col})

    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))

def handle_prep_custom(args):
    df = pd.read_csv(args.file)

    df['Province/State'] = df['Province/State'].fillna('NA_' + df['Country/Region'])

    df['Date'] = pd.to_datetime(df['Date'])

    basedate = pd.Timestamp('2020-01-22')

    df['days_since'] = (df['Date'] - basedate).dt.days

    # Taken from https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons
    world_pop = pd.read_csv('data/population_by_country_2020.csv')
    world_pop.columns = ['Country/Region', 'Population']

    df = pd.merge(df, world_pop, on='Country/Region', how='left')

    cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia', 
            'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines', 
            'Taiwan*', 'US', 'West Bank and Gaza', 'Sao Tome and Principe']
    pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000, 
            53109, 110854, 23806638, 330541757, 4543126, 219159]

    for c, p in zip(cols, pops):
        df.loc[df['Country/Region']== c, 'Population'] = p
        
    df['Confirmed / Million Inhabitants'] = round((df['Confirmed'] / df['Population']) * 1000000)
    df['Deaths / Million Inhabitants'] = round((df['Deaths'] / df['Population']) * 1000000)
    df['Recovered / Million Inhabitants'] = round((df['Recovered'] / df['Population']) * 1000000)

    df = df[df['Population'].notna()]

    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))

def handle_prep_date_between(args):
    df = pd.read_csv(args.file)

    df['Date'] = pd.to_datetime(df['Date'])

    first_date = datetime.strptime(args.first_date, '%Y-%m-%d')
    second_date = datetime.strptime(args.second_date, '%Y-%m-%d')

    df = df.loc[(df['Date'] >= first_date) & (df['Date'] <= second_date)]

    print(df)

    if args.out_file:
        df.to_csv(args.out_file, index=False)
        print('Output saved in {0} file'.format(args.out_file))