from collections import namedtuple
import configparser
from ast import literal_eval
import os

def parse_ini(config_path: str):
    read_config = configparser.ConfigParser()
    read_config.read(config_path)
    config_attribs = []
    data_dict = {}
    for section in read_config.sections():
        for (key, value) in read_config.items(section):
            config_attribs.append(key)
            if value.replace('.', '', 1).replace('+', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit():
                # Exponential format and decimal format should be accounted for
                data_dict[key] = literal_eval(value)
            elif value == 'True' or value == 'true' or value == 'False' or value == 'false':
                if value == 'True' or value == 'true':
                    data_dict[key] = True
                else:
                    data_dict[key] = False
            elif value == 'None':
                data_dict[key] = None
            elif ',' in value:  # Config contains lists
                if ', ' in value:
                    is_number = any(char.isdigit() for char in value.split(', ')[0])
                    items_list = value.split(', ')
                    if '' in items_list:
                        items_list.remove('')
                    if is_number:
                        data_dict[key] = [literal_eval(val) for val in items_list]
                    else:
                        data_dict[key] = [val for val in items_list]
                else:
                    is_number = any(char.isdigit() for char in value.split(',')[0])
                    items_list = value.split(',')
                    if '' in items_list:
                        items_list.remove('')
                    if is_number:
                        data_dict[key] = [literal_eval(val) for val in items_list]
                    else:
                        data_dict[key] = [val for val in items_list]
            else:
                data_dict[key] = value

    Config = namedtuple('Config', config_attribs)
    cfg = Config(**data_dict)
    return cfg


def parse_value(value):
    if value.replace('.', '', 1).replace('+', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit():
        # Exponential format and decimal format should be accounted for
        return literal_eval(value)
    elif value == 'True' or value == 'False':
        if value == 'True':
            return True
        else:
            return False
    elif value == 'None':
        return None
    elif ',' in value:  # Config contains lists
        is_number = any(char.isdigit() for char in value.split(',')[0])
        items_list = value.split(',')

        if '' in items_list:
            items_list.remove('')
        if is_number:
            return [literal_eval(val) for val in items_list]
        else:
            if '\"' in items_list[0] and '\'' in items_list[0]:
                return [literal_eval(val.strip()) for val in items_list]
            else:
                return [val.strip() for val in items_list]
    else:
        return value


def save_ini(config_path, log_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    save_path = os.path.join(log_path, 'config.ini')
    with open(save_path, 'w') as f:
        config.write(f)

