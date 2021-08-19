"""Top-level package for COSC522."""

from cosc522.fancy_logger import ColorizedLogger
from cosc522.timing_tools import timeit
from cosc522.profiling_funcs import profileit
from cosc522.configuration import Configuration, validate_json_schema
from cosc522.cloudstore import DropboxCloudstore
from cosc522.datastore import MySqlDatastore
from cosc522.emailer import GmailEmailer

__author__ = "drkostas"
__email__ = "georgiou.kostas94@gmail.com"
__version__ = "0.1.0"
