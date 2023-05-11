import os
import sys
import copy
from time import time
import re
import math
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import mpl_interactions.ipyplot as iplt
from PIL import Image
import shutil
from scipy import interpolate


a = np.array([[1,2], [3,4]])

print(a)
a-=1
print(a)