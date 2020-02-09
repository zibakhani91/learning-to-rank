# %%

import sys

sys.path.insert(0, '../lib')  # noqa
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn.metrics import *
% matplotlib
inline
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import csv
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from greedy_order import *

# %% md

# Load dataset

# %%

genres_data = pd.read_csv(
    'movielens-dataset/u.genre',
    sep='|',
    encoding="ISO-8859-1",
    header=None,
    names=['name', 'id']
)

# %%

movie_data_columns = np.append(
    ['movie_id', 'title', 'release_date', 'video_release_date', 'url'],
    genres_data['name'].values
)

# %%

movie_data = pd.read_csv(
    'movielens-dataset/u.item',
    sep='|',
    encoding="ISO-8859-1",
    header=None,
    names=movie_data_columns,
    index_col='movie_id'
)

# %%

selected_columns = np.append(['title', 'release_date'], genres_data['name'].values)
movie_data = movie_data[selected_columns]
movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])

movie_data.head()

# %%

ratings_data = pd.read_csv(
    'movielens-dataset/u.data',
    sep='\t',
    encoding="ISO-8859-1",
    header=None,
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# %%

movie_data['ratings_average'] = ratings_data.groupby(['movie_id'])['rating'].mean()
movie_data['ratings_count'] = ratings_data.groupby(['movie_id'])['rating'].count()

# %%

movie_data[['title', 'ratings_average', 'ratings_count']].head()

# %% md

# Remove null values

# %%

movie_data[selected_columns].isnull().any()

# %%

null_release_dates = movie_data[movie_data['release_date'].isnull()]
assert null_release_dates.shape[0] == 1

# %%

movie_data = movie_data.drop(null_release_dates.index.values)
assert movie_data[selected_columns].isnull().any().any() == False

# %% md

# Check data types

# %%

movie_data.dtypes

# %% md

# Compute the artificial "price" and "buy_probability" attributes

# %%

from datetime import datetime
import dateutil

# %%

oldest_date = pd.to_datetime(movie_data['release_date']).min()
most_recent_date = pd.to_datetime(movie_data['release_date']).max()
normalised_age = (most_recent_date - pd.to_datetime(movie_data['release_date'])) / (most_recent_date - oldest_date)
normalised_rating = (5 - movie_data['ratings_average']) / (5 - 1)

movie_data['price'] = np.round((1 - normalised_rating) * (1 - normalised_age) * 10)
movie_data[['title', 'price', 'ratings_average', 'ratings_count']].head()

# %%

# one movie had title unknown, relesease data unknown, etc...
movie_data = movie_data[movie_data['price'].notnull()]

# %%

# the lower the price, the more likely I am going to buy
movie_data['buy_probability'] = 1 - movie_data['price'] * 0.1

# %% md

# The perfect ranking

# %%

plt.plot(movie_data['price'].values, movie_data['buy_probability'].values, 'ro')  # ro = red circles
plt.xlabel('price')
plt.ylabel('buy_probability')
plt.show()


# %% md

## Genres distribution

# %%

def plot_genres(movie_data):
    genres_array = [(genre, movie_data[genre].sum()) for genre in genres_data['name'].values]
    genres_names = list(map(lambda x: x[0], genres_array))
    genres_count = list(map(lambda x: x[1], genres_array))

    genres_df = pd.DataFrame({'name': genres_names, 'count': genres_count})
    x = np.arange(len(genres_df['name'].values))
    plt.bar(x, genres_df['count'].values)
    plt.xticks(x, genres_df['name'].values, rotation='vertical')
    plt.title("Genres distribution")


def plot_years(movie_data):
    years = movie_data['release_date'].values.astype('datetime64[Y]').astype(int) + 1970
    plt.hist(years)
    plt.title("Years distribution")


def plot_ratings(movie_data):
    movie_data['ratings_average'].hist()
    plt.title("Ratings distribution")


def plot_price(movie_data):
    movie_data['price'].hist()
    plt.title("Price distribution")


def plot_ratings_count(movie_data):
    movie_data['ratings_count'].hist()
    plt.title("Ratings count")


def buy_probability_distribution(movie_data):
    movie_data['buy_probability'].hist(bins=11)
    plt.title("Buy probability distribution")


# %%

def plot_distributions(movie_data):
    fig = plt.figure(figsize=(8, 6))

    fig.add_subplot(3, 2, 1)
    plot_genres(movie_data)

    fig.add_subplot(3, 2, 2)
    plot_years(movie_data)

    fig.add_subplot(3, 2, 3)
    plot_ratings(movie_data)

    fig.add_subplot(3, 2, 4)
    plot_price(movie_data)

    fig.add_subplot(3, 2, 5)
    plot_ratings_count(movie_data)

    fig.add_subplot(3, 2, 6)
    buy_probability_distribution(movie_data)

    plt.tight_layout()


# %%

plot_distributions(movie_data)


# %% md

# Events database functions

# %%

# The users database
class User:
    def __init__(self, id):
        self.id = id
        self.positive = []
        self.negative = []

    def add_positive(self, movie_id):
        self.positive.append(movie_id)

    def add_negative(self, movie_id):
        self.negative.append(movie_id)

    def get_positive(self):
        return self.positive

    def get_negative(self):
        return self.negative


# %%

np.random.seed(1)


class EventsGenerator:
    NUM_OF_OPENED_MOVIES_PER_USER = 20
    NUM_OF_USERS = 1000

    def __init__(self, learning_data, buy_probability):
        self.learning_data = learning_data
        self.buy_probability = buy_probability
        self.users = []
        for id in range(1, self.NUM_OF_USERS):
            self.users.append(User(id))

    def run(self, pairwise=False):
        for user in self.users:
            opened_movies = np.random.choice(self.learning_data.index.values, self.NUM_OF_OPENED_MOVIES_PER_USER)
            self.__add_positives_and_negatives_to(user, opened_movies)

        if pairwise:
            return self.__build_pairwise_events_data()
        else:
            return self.__build_events_data()

    def __add_positives_and_negatives_to(self, user, opened_movies):
        for movie_id in opened_movies:
            if np.random.binomial(1, self.buy_probability.loc[movie_id]):
                user.add_positive(movie_id)
            else:
                user.add_negative(movie_id)

    def __build_events_data(self):
        events_data = []

        for user in self.users:
            for positive_id in user.get_positive():
                tmp = learning_data.loc[positive_id].to_dict()
                tmp['outcome'] = 1
                events_data += [tmp]

            for negative_id in user.get_negative():
                tmp = learning_data.loc[negative_id].to_dict()
                tmp['outcome'] = 0
                events_data += [tmp]

        return pd.DataFrame(events_data)

    def __build_pairwise_events_data(self):
        events_data = []

        for i, user in enumerate(self.users):
            print("{} of {}".format(i, len(self.users)))
            positives = user.get_positive()
            negatives = user.get_negative()

            sample_size = min(len(positives), len(negatives))

            positives = np.random.choice(positives, sample_size)
            negatives = np.random.choice(negatives, sample_size)

            # print("Adding {} events".format(str(len(positives) * len(negatives) * 2)))
            for positive in positives:
                for negative in negatives:
                    e1 = learning_data.loc[positive].values
                    e2 = learning_data.loc[negative].values

                    pos_neg_example = np.concatenate([e1, e2, [1]])
                    neg_pos_example = np.concatenate([e2, e1, [0]])

                    events_data.append(pos_neg_example)
                    events_data.append(neg_pos_example)

        c1 = [c + '_1' for c in learning_data.columns]
        c2 = [c + '_2' for c in learning_data.columns]
        return pd.DataFrame(events_data, columns=np.concatenate([c1, c2, ['outcome']]))


# %%

def build_learning_data_from(movie_data):
    feature_columns = np.setdiff1d(movie_data.columns, np.array(['title', 'buy_probability']))
    learning_data = movie_data.loc[:, feature_columns]

    scaler = StandardScaler()
    learning_data.loc[:, ('price')] = scaler.fit_transform(learning_data[['price']])
    learning_data['ratings_average'] = scaler.fit_transform(learning_data[['ratings_average']])
    learning_data['ratings_count'] = scaler.fit_transform(learning_data[['ratings_count']])
    learning_data['release_date'] = learning_data['release_date'].apply(lambda x: x.year)
    learning_data['release_date'] = scaler.fit_transform(learning_data[['release_date']])

    return learning_data


# %%

def plot_events_distribution(events_data):
    events_data_sample = events_data.sample(frac=0.1)
    negative_outcomes = events_data_sample[events_data_sample['outcome'] == 0.0]['price']
    positive_outcomes = events_data_sample[events_data_sample['outcome'] == 1.0]['price']

    outcomes = np.array(list(zip(negative_outcomes.values, positive_outcomes.values)))
    plt.hist(outcomes, bins=11, label=['Negative', 'Positive'])
    plt.legend()
    plt.xlabel('price')
    plt.show()


# %%

def get_feature_columns_from(learning_data, pairwise=False):
    if not pairwise:
        return learning_data.columns.values
    else:
        f1 = [c + '_1' for c in learning_data.columns.values]
        f2 = [c + '_2' for c in learning_data.columns.values]
        f1.extend(f2)
        return np.asarray(f1)


# %%

def save_events_data(events_data, learning_data, tag, pairwise=False):
    events_data = events_data.reindex(np.random.permutation(events_data.index))
    events_data.to_csv('movie_events_' + tag + '.csv')

    if not pairwise:
        df = pd.DataFrame(get_feature_columns_from(learning_data))
        df.to_csv("feature_columns_" + tag + ".csv")
    else:
        df = pd.DataFrame(get_feature_columns_from(learning_data, pairwise=True))
        df.to_csv("feature_columns_" + tag + ".csv")


# %%

def load_events_data(tag):
    events_data = pd.DataFrame.from_csv('movie_events_' + tag + '.csv')
    tmp = pd.DataFrame.from_csv("feature_columns_" + tag + ".csv")
    feature_columns = tmp['0'].values

    return [events_data, feature_columns]


# %%

def get_test_train_data(events_data, feature_columns):
    X = events_data.loc[:, feature_columns].values.astype(np.float32)
    print('overall input shape: ' + str(X.shape))

    y = events_data.loc[:, ['outcome']].values.astype(np.float32).ravel()
    print('overall output shape: ' + str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('training input shape: ' + str(X_train.shape))
    print('training output shape: ' + str(y_train.shape))

    print('testing input shape: ' + str(X_test.shape))
    print('testing output shape: ' + str(y_test.shape))

    return [X_train, X_test, y_train, y_test]


# %% md

# Generate linear events

# %%

learning_data = build_learning_data_from(movie_data)

# %%

events_data = EventsGenerator(learning_data, movie_data['buy_probability']).run()

# %%

save_events_data(events_data, learning_data, 'linear')

# %%

events_data, feature_columns = load_events_data('linear')

# %%

plot_events_distribution(events_data)

# %% md

## Pairwise

# %%

events_data = EventsGenerator(learning_data, movie_data['buy_probability']).run(pairwise=True)

# %%

save_events_data(events_data, learning_data, 'pairwise-linear', pairwise=True)

# %%

events_data, feature_columns = load_events_data('pairwise-linear')

# %%

events_data.shape

# %% md

# Train/Test data split

# %%

X_train, X_test, y_train, y_test = get_test_train_data(events_data, feature_columns)


# %% md

# Utility functions

# %%

def plot_rank(features, model, learning_data, predict_fun):
    lg_input = learning_data.values.astype(np.float32)
    print('overall input shape: ' + str(lg_input.shape))

    learning_data_with_rank = learning_data.copy()
    learning_data_with_rank['rank'] = predict_fun(model, lg_input)

    for idx, feature in enumerate(features):
        plt.subplot(len(features), 1, idx + 1)
        plt.plot(learning_data_with_rank[feature].values, learning_data_with_rank['rank'].values, 'ro')
        plt.xlabel(feature)
        plt.ylabel('rank')

    plt.tight_layout()
    plt.show()


# %%

def train_model(model, prediction_function, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_train_pred = prediction_function(model, X_train)

    print('train precision: ' + str(precision_score(y_train, y_train_pred)))
    print('train recall: ' + str(recall_score(y_train, y_train_pred)))
    print('train accuracy: ' + str(accuracy_score(y_train, y_train_pred)))

    y_test_pred = prediction_function(model, X_test)

    print('test precision: ' + str(precision_score(y_test, y_test_pred)))
    print('test recall: ' + str(recall_score(y_test, y_test_pred)))
    print('test accuracy: ' + str(accuracy_score(y_test, y_test_pred)))

    return model


# %% md

# Rank with the perfect predictor

# %%

def get_predicted_outcome(model, data):
    return np.rint(model.predict(data))


# %%

def get_predicted_rank(model, data):
    return model.predict(data)


# %%

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# %%

class PerfectPredictor:
    def fit(self, X, y):
        return None

    def predict(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        return 1 - min_max_scaler.fit_transform(X[:, -5])


# %%

model = train_model(PerfectPredictor(), get_predicted_outcome, X_train, y_train, X_test, y_test)
plot_rank(['price'], model, learning_data, get_predicted_rank)

# %% md

# Rank with a Logistic Regression

*Collect
for each movie the buy probability from the raw events
*Run
a
beta
regression

= > Expect
the
coefficients
to
represent
the
artificial
probability
function


# %%

def get_predicted_outcome(model, data):
    return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)


# %%

def get_predicted_rank(model, data):
    return model.predict_proba(data)[:, 1]


# %%

model = train_model(LogisticRegression(), get_predicted_outcome, X_train, y_train, X_test, y_test)

# %%

plot_rank(['price'], model, learning_data, get_predicted_rank)


# %% md

# Rank with Neural Network

# %%

def nn():
    return NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 23),  # this code won't compile without SIZE being set
        hidden_num_units=46,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=1,  # this code won't compile without OUTPUTS being set

        # optimization method:
        update_learning_rate=0.01,
        regression=True,  # If you're doing classification you want this off
        max_epochs=50,  # more epochs can be good,
        verbose=1,  # enabled so that you see meaningful output when the program runs
    )


# %%

def get_predicted_outcome(model, data):
    return np.rint(model.predict(data))


# %%

def get_predicted_rank(model, data):
    return model.predict(data)


# %%

model = train_model(
    nn(),
    get_predicted_outcome,
    X_train.astype(np.float32),
    y_train.astype(np.float32),
    X_test.astype(np.float32),
    y_test.astype(np.float32)
)

# %%

plot_rank(['price'], model, learning_data, get_predicted_rank)


# %% md

# Rank with Decision Trees

# %%

def get_predicted_outcome(model, data):
    return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)


# %%

def get_predicted_rank(model, data):
    return model.predict_proba(data)[:, 1]


# %%

from sklearn import tree

model = train_model(tree.DecisionTreeClassifier(), get_predicted_outcome, X_train, y_train, X_test, y_test)

# %%

plot_rank(['price'], model, learning_data, get_predicted_rank)

# %% md

# Customers with non-linear buying behaviour

# %%

price_component = np.sqrt(movie_data['price'] * 0.1)
ratings_component = np.sqrt(movie_data['ratings_average'] * 0.1 * 2)
movie_data['buy_probability'] = 1 - price_component * 0.2 - ratings_component * 0.8

# %%

plot_distributions(movie_data)

# %%

plt.subplot(2, 1, 1)
plt.plot(movie_data['price'].values, movie_data['buy_probability'].values, 'ro')  # ro = red circles
plt.xlabel('price')
plt.ylabel('buy_probability')

plt.subplot(2, 1, 2)
plt.plot(movie_data['ratings_average'].values, movie_data['buy_probability'].values, 'ro')  # ro = red circles
plt.xlabel('ratings_average')
plt.ylabel('buy_probability')

plt.tight_layout()
plt.show()

# %% md

# Create events

# %%

learning_data = build_learning_data_from(movie_data)

# %%

positive_events, negative_events = generate_events()
events_data = build_events_data(positive_events, negative_events, learning_data)
save_events_data(events_data, learning_data, 'nonlinear')

# %%

events_data, feature_columns = load_events_data('nonlinear')

# %%

plot_events_distribution(events_data)

# %%

X_train, X_test, y_train, y_test = get_test_train_data(events_data)


# %% md

# Rank with the perfect predictor

# %%

def get_predicted_outcome(model, data):
    return np.rint(model.predict(data))


# %%

def get_predicted_rank(model, data):
    return model.predict(data)


# %%

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# %%

class PerfectPredictor:
    def fit(self, X, y):
        return None

    def predict(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        price_component = np.sqrt(min_max_scaler.fit_transform(X[:, -5]))
        ratings_component = np.sqrt(min_max_scaler.fit_transform(X[:, -4]))
        return 1 - price_component * 0.2 - ratings_component * 0.8


# %%

model = train_model(PerfectPredictor(), get_predicted_outcome, X_train, y_train, X_test, y_test)
plot_rank(['price', 'ratings_average'], model, learning_data, get_predicted_rank)


# %% md

# Rank with Logistic Regression

# %%

def get_predicted_outcome(model, data):
    return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)


# %%

def get_predicted_rank(model, data):
    return model.predict_proba(data)[:, 1]


# %%

model = train_model(LogisticRegression(), get_predicted_outcome, X_train, y_train, X_test, y_test)

# %%

plot_rank(['price', 'ratings_average'], model, learning_data, get_predicted_rank)


# %% md

# Rank with Neural Networks

# %%

def get_predicted_outcome(model, data):
    return np.rint(model.predict(data))


# %%

def get_predicted_rank(model, data):
    return model.predict(data)


# %%

model = train_model(
    nn(),
    get_predicted_outcome,
    X_train.astype(np.float32),
    y_train.astype(np.float32),
    X_test.astype(np.float32),
    y_test.astype(np.float32)
)

# %%

plot_rank(['price', 'ratings_average'], model, learning_data, get_predicted_rank)


# %% md

# Rank with Decision Trees

# %%

def get_predicted_outcome(model, data):
    return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)


# %%

def get_predicted_rank(model, data):
    return model.predict_proba(data)[:, 1]


# %%

from sklearn import tree

model = train_model(tree.DecisionTreeClassifier(), get_predicted_outcome, X_train, y_train, X_test, y_test)

# %%

plot_rank(['price', 'ratings_average'], model, learning_data, get_predicted_rank)


# %% md

# Pairwise learning

# %%

def nn():
    return NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 46),  # this code won't compile without SIZE being set
        hidden_num_units=92,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=1,  # this code won't compile without OUTPUTS being set

        # optimization method:
        update_learning_rate=0.01,
        regression=True,  # If you're doing classification you want this off
        max_epochs=50,  # more epochs can be good,
        verbose=1,  # enabled so that you see meaningful output when the program runs
    )


# %%

def get_predicted_outcome(model, data):
    return np.rint(model.predict(data))


# %%

def plot_rank_for_pairwise(features, model, learning_data, predict_fun):
    learning_data_with_rank = learning_data.copy()
    learning_data_with_rank = predict_fun(model, learning_data_with_rank)

    for idx, feature in enumerate(features):
        plt.subplot(len(features), 1, idx + 1)
        plt.plot(learning_data_with_rank[feature].values, learning_data_with_rank['rank'].values, 'ro')
        plt.xlabel(feature)
        plt.ylabel('rank')

    plt.tight_layout()
    plt.show()


# %%

def get_predicted_rank(model, data):
    cached_preference = {}

    def preference(x, y):
        if cached_preference.get((x, y)) is not None:
            return cached_preference[x, y]

        x_v = data.loc[x].values
        y_v = data.loc[y].values
        cached_preference[x, y] = model.predict(np.array([np.concatenate([x_v, y_v])]))
        return cached_preference[x, y]

    g = GreedyOrder(data.index, preference)
    order = g.run()
    for i, row in data.iterrows():
        data.loc[i, 'rank'] = 1 - (order.index(i) / float(len(order)))

    return data


# %%

from importlib import reload
import greedy_order

reload(greedy_order)
from greedy_order import *

# %%

model = train_model(
    nn(),
    get_predicted_outcome,
    X_train.astype(np.float32),
    y_train.astype(np.float32),
    X_test.astype(np.float32),
    y_test.astype(np.float32)
)

# %%

plot_rank_for_pairwise(['price'], model, learning_data, get_predicted_rank)

# %%


