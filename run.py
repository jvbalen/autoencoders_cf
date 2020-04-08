from argparse import ArgumentParser

import gin

from models import *
from data import DataLoader
from preprocessing import preprocess


@gin.configurable
def experiment(data_path, Recommender=LinearRecommender, log_dir=None, cap=None):

    print('Loading data...')
    loader = DataLoader(data_path)
    x_train = loader.load_data('train')
    x_val, y_val = loader.load_data('validation')

    print('Preprocessing...')
    if cap:
        x_train = x_train[:cap]
        x_val = x_val[:cap]
        y_val = y_val[:cap]
    x_train, y_train, x_val, y_val = preprocess(x_train, x_train.copy(), x_val, y_val)

    print('Training...')
    recommender = Recommender(log_dir=log_dir)
    metrics = recommender.train(x_train, y_train, x_val, y_val)

    return metrics


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='directory containing pre-processed dataset', type=str)
    parser.add_argument('--logdir', help='log directory', type=str)
    parser.add_argument('--config', help='path to gin config file', type=str, default=None)
    args = parser.parse_args()

    # override keyword arguments in gin.configurable modules from config file
    if args.config:
        gin.parse_config_file(args.config)

    metrics = experiment(args.data, log_dir=args.logdir)
    print(metrics)
