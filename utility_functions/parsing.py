import argparse


def get_supervised_parser():
    parser = argparse.ArgumentParser(description='MNN supervised experimental setup')
    parser.add_argument('-k',
                        '--number_folders',
                        type=int,
                        default=10,
                        help='The number k of folds for cross-validation.')
    parser.add_argument('-e',
                        '--number-epochs',
                        type=int,
                        default=50,
                        help='The number of train epochs per fold.')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=1e-3,
                        help='The learning rate for the Adam gradient descent optimizer.')

    parser.add_argument('--hybrid',
                        action='store_true',
                        help='Use hybrid training. Script defaults to this mode.')
    parser.add_argument('--no-hybrid',
                        dest='hybrid',
                        action='store_false',
                        help='Use static training. Overrides default hybrid training.')

    parser.set_defaults(hybrid=True)
    return parser


def get_fine_tune_parser(default_model_dir: str):
    parser = argparse.ArgumentParser(description='MNN fine-tune experimental setup')
    parser.add_argument('-k',
                        '--number_folders',
                        type=int,
                        default=10,
                        help='The number k of folds for cross-validation.')
    parser.add_argument('-e',
                        '--number-epochs',
                        type=int,
                        default=50,
                        help='The number of train epochs per fold.')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=1e-3,
                        help='The learning rate for the Adam gradient descent optimizer.')
    parser.add_argument('-d',
                        '--model_directory',
                        type=str,
                        default=default_model_dir,
                        help='The relative path where the MNN\'s pre-trained weights are stored.')

    parser.add_argument('--ph16',
                        action='store_true',
                        help='Fine-tune to PhysioNet\'16.')
    parser.add_argument('--circor22',
                        dest='ph16',
                        action='store_false',
                        help='Fine-tune to CirCor\'22.')

    parser.add_argument('--hybrid',
                        action='store_true',
                        help='Use hybrid training. Script defaults to this mode.')
    parser.add_argument('--no-hybrid',
                        dest='hybrid',
                        action='store_false',
                        help='Use static training. Overrides default hybrid training.')

    parser.set_defaults(hybrid=True)
    return parser