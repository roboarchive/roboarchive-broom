#!/usr/bin/env python3
import argparse
import datetime
import os

import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from cnn import get_cnn

from data_generator import DataSource


def configure_backend(args):
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    if args.cpu:
        config.device_count['GPU'] = 0
    sess = tf.Session(config=config)
    K.set_session(sess)


def train(args):
    configure_backend(args)

    # np.random.seed(123)  # for reproducibility
    print('Creating CNN')

    cnn = get_cnn(args)
    model_checkpoint = ModelCheckpoint(
        args.weights_file,
        monitor=args.monitor,
        verbose=1,
        save_best_only=args.best,
        period=args.period,
    )
    if args.comment:
        uniq_part = args.comment
    else:
        uniq_part = f'lr_{args.learning_rate}'
    run_name = datetime.datetime.now().strftime(f'%Y/%m/%d/%H_%M_{uniq_part}')
    callbacks = [model_checkpoint, TensorBoard(log_dir=f'tensorboard_log/{run_name}')]
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', verbose=1, patience=50))
    data_source = DataSource(args, cnn)
    steps = args.epoch_steps if args.epoch_steps else data_source.ideal_steps
    print(f'Number of Steps: {steps} / {max(int(steps / args.batch_size), 1)}')
    cnn.model.fit_generator(
        data_source.data_generator(),
        validation_data=data_source.validation_generator(),
        validation_steps=max(int(steps / args.batch_size), 1),
        steps_per_epoch=steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )


def add_common_arguments(parser):
    parser.add_argument(
        '-w', '--weights', dest='weights_file', help='Save weights to file', default='weights.h5'
    )
    parser.add_argument(
        '-c',
        '--cnn',
        dest='cnn_name',
        choices=['simple', 'unet'],
        help='CNN',
        default=os.environ.get('CNN_NAME') or 'unet',
    )
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--tile-size', default=256, type=int)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--comment')
    parser.add_argument('--monitor', default='val_loss')
    parser.add_argument('--period', default=1, type=int)
    parser.add_argument('-e', '--epochs', default=100000, type=int)
    parser.add_argument('--epoch-steps', default=None, type=int)
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-f', '--filter')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--transformated', type=float, default=0.5,
                        help='Ratio of transoformated images')
    parser.add_argument('--no-generated', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
