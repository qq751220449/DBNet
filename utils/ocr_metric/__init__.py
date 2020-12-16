from .icdar2015 import QuadMetric


def get_metric(config):
    try:
        if 'args' not in config:
            args = {}
        else:
            args = config['args']
        if isinstance(args, dict):
            cls = eval(config['type'])(**args)
        else:
            cls = eval(config['type'])(args)
        return cls
    except:
        return None