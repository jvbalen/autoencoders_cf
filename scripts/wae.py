"""
Ex.:
python wae.py --data ~/data/ml-20m/ --logdir ~/experiments/wae/logs/WAE/ \
    --config config/001.gin
"""
import gin
from argparse import ArgumentParser

from data import DataLoader
from preprocessing import preprocess
from models.tf import build_model, train


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data', help='directory containing pre-processed dataset', type=str)
    parser.add_argument('--cap', help='cap train/val data for debugging', type=int, default=None)
    parser.add_argument('--logdir', help='log directory for tensorboard', type=str)
    parser.add_argument('--config', help='path to gin config file', type=str)
    args = parser.parse_args()

    # override keyword arguments in gin.configurable modules from config file
    gin.parse_config_file(args.config)

    print('Loading data...')
    loader = DataLoader(args.data)
    x_train = loader.load_data('train')
    x_val, y_val = loader.load_data('validation')

    print('Constructing model...')
    model = build_model(x_train)  # don't cap yet, we want realistic sparsities

    print('Preprocessing...')
    if args.cap:
        x_train = x_train[:args.cap]
        x_val, y_val = x_val[:args.cap], y_val[:args.cap]
    x_train, y_train, x_val, y_val = preprocess(x_train, x_train.copy(), x_val, y_val)

    print('Training...')
    train(model, x_train, y_train, x_val, y_val, log_dir=args.logdir)
