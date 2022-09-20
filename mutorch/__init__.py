import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from core import *
from core import nn as nn
from core import optim as optim
from core import losses as losses