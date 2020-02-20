from argparse import ArgumentParser

import gin

from models.skl import SKLRecommender
from data import DataLoader
from preprocessing import preprocess


def run_experiment(data_path, log_dir, Recommender=SKLRecommender, config_path=None, cap=None):

    # override keyword arguments in gin.configurable modules from config file
    if config_path:
        gin.parse_config_file(config_path)

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
    recommender = Recommender(log_dir)
    metrics = recommender.train(x_train, y_train, x_val, y_val)

    return metrics


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='directory containing pre-processed dataset', type=str)
    parser.add_argument('--cap', help='cap train/val data for debugging', type=int, default=None)
    parser.add_argument('--logdir', help='log directory', type=str)
    parser.add_argument('--config', help='path to gin config file', type=str, default=None)
    args = parser.parse_args()

    metrics = run_experiment(args.data. args.logdir, config_path=args.config, cap=args.cap)
    print(metrics)
