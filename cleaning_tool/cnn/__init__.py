from .simple_cnn import SimpleCNN
from .unet import UnetCNN


def get_cnn(args):
    if args.cnn_name == 'simple':
        return SimpleCNN(args)
    elif args.cnn_name == 'unet':
        return UnetCNN(args)
    else:
        if not args.cnn_name:
            raise Exception(f'You should specify cnn_name via: -c/-cnn or CNN_NAME env variable')
        raise Exception('unknown name {}'.format(args.cnn_name))
