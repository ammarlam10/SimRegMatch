import argparse


def SimRegMatch_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', type=str, default='SimRegMatch')

    # For data
    parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb', 'utkface', 'so2sat_pop'], help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--labeled-ratio', type=float, default=0.1)
    parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
    parser.add_argument('--normalize-labels', action='store_true', default=False, help='Normalize labels using mean/std from training data (recommended for large-scale targets like population)')
    parser.add_argument('--log-transform', action='store_true', default=False, help='Apply log1p transform to labels (recommended for highly skewed targets like population)')

    # For model architecture
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='resnet50', help='model name')
    
    # For uncertainty estimation
    parser.add_argument('--threshold', type=float, default=10)
    parser.add_argument('--percentile', type=float, default=0.95)    
    parser.add_argument('--iter-u', type=int, default=5)
    
    # For pseudo-label calibration
    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.6)

    # For loss calculation
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
    parser.add_argument('--huber-delta', type=float, default=1.0, help='Delta parameter for Huber loss (errors < delta use L2, errors > delta use L1)')
    parser.add_argument('--lambda-u', type=float, default=0.01)
    
    # For model training
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')    
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='optimizer weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    # seed
    parser.add_argument('--seed', default=0)
    
    # GPU device (default 0 since only one GPU will be exposed to container)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (0 when only one GPU exposed)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=1, help='Starting epoch number (used when resuming)')
    
    return parser
