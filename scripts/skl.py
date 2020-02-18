from argparse import ArgumentParser

import gin

from models.skl import build_model, evaluate, coefs_from_model
from data import DataLoader
from util import Logger


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='directory containing pre-processed dataset', type=str)
    parser.add_argument('--cap', help='cap train/val data for debugging', type=int, default=None)
    parser.add_argument('--logdir', help='log directory', type=str)
    parser.add_argument('--config', help='path to gin config file', type=str, default=None)
    args = parser.parse_args()

    # override keyword arguments in gin.configurable modules from config file
    if args.config:
        gin.parse_config_file(args.config)
    logger = Logger(args.logdir)

    print('Loading data...')
    loader = DataLoader(args.data)
    x_train = loader.load_data('train')
    x_val, y_val = loader.load_data('validation')

    print('Constructing model...')
    model = build_model()

    print('Training...')
    if args.cap:
        x_train = x_train[:args.cap]
        x_val = x_val[:args.cap]
        y_val = y_val[:args.cap]
    model.fit(x_train, x_train.toarray() > 0.0)

    print('Evaluating...')
    metrics = evaluate(model, x_val, y_val)
    logger.log_config(gin.operative_config_str())
    logger.log_results(metrics, config=gin.operative_config_str())
    logger.log_coefs(*coefs_from_model(model))
