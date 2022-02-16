__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
"""Logging functions.

"""
import logging

logger = logging.getLogger("segar")


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


def set_logger(debug=False):
    global logger
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s:%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    dup_filter = DuplicateFilter()
    handler.addFilter(dup_filter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
