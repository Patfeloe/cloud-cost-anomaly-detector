import json


import numpy as np
import pandas as pd

AnomalyType = Literal["spike", "drop", "sustained_increase"]

SERVICES_CATEGORIES