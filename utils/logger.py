import logging
import pprint


class Logger(object):
    def __init__(self, filename='./log.txt', level=logging.INFO):
        logging.basicConfig(level=level, filename=filename,
                            format='%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler = logging.StreamHandler()
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)
        self.logger = logger

    def get_logger(self):
        return self.logger


def log_args(filename, args):
    with open(filename, 'a') as out:
        pprint.pprint(vars(args), stream=out)
