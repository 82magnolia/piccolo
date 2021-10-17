from parse_utils import parse_ini, save_ini, parse_value
import argparse
import localize
import os
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import configparser


if __name__ == '__main__':
    # General config parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file to use for running experiments", default=None, type=str)
    parser.add_argument("--log", help="Log directory for logging accuracy", default="./log", type=str)
    parser.add_argument('--override', default=None, help='Arguments for overriding config')
    args = parser.parse_args()
    cfg = parse_ini(args.config)

    log_dir = args.log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    if args.override is not None:
        equality_split = args.override.split('=')
        num_equality = len(equality_split)
        assert num_equality > 0
        if num_equality == 2:
            override_dict = {equality_split[0]: parse_value(equality_split[1])}
        else:
            keys = [equality_split[0]]  # First key
            keys += [equality.split(',')[-1] for equality in equality_split[1:-1]]  # Other keys
            values = [equality.replace(',' + key, '') for equality, key in zip(equality_split[1:-1], keys[1:])]  # Get values other than last field
            values.append(equality_split[-1])  # Get last value
            values = [value.replace('[', '').replace(']', '') for value in values]

            override_dict = {key: parse_value(value) for key, value in zip(keys, values)}

        cfg_dict = cfg._asdict()

        Config = namedtuple('Config', tuple(set(cfg._fields + tuple(override_dict.keys()))))
        
        cfg_dict.update(override_dict)

        cfg = Config(**cfg_dict)

    config = configparser.ConfigParser()
    config.add_section('Default')

    cfg_dict = cfg._asdict()

    for key in cfg_dict:
        if key != 'name':
            config['Default'][key] = str(cfg_dict[key]).replace('[', '').replace(']', '')
        else:
            config['Default'][key] = str(cfg_dict[key])

    with open(os.path.join(args.log, 'config.ini'), 'w') as configfile:
        config.write(configfile)

    # Branch on dataset
    dataset = cfg.dataset
    if dataset == 'Stanford2D-3D-S':
        localize.localize_stanford(cfg, writer, log_dir)
    else:
        raise ValueError
