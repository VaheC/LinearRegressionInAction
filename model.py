# Importing necessary packages
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Creating a logger function to log ML model building tasks
def create_logger(log_file):
	'''Creates a logger

	   Parameters:
	   log_file: str, name of a log file to create
	   Result:
	   logger: a logger which outputs logging to a file and console
	'''
	if os.path.isfile(f'{log_file}.log'):
		os.remove(f'{log_file}.log')

	logging.basicConfig(level=logging.INFO, filemode='w')	 
	logger = logging.getLogger(__name__)

	c_handler = logging.StreamHandler()
	f_handler = logging.FileHandler(f'{log_file}.log')

	c_handler.setLevel(logging.INFO)
	f_handler.setLevel(logging.INFO)

	c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
								 datefmt='%d-%b-%y %H:%M:%S')
	f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
								 datefmt='%d-%b-%y %H:%M:%S')
	c_handler.setFormatter(c_format)
	f_handler.setFormatter(f_format)

	logger.addHandler(c_handler)
	logger.addHandler(f_handler)

	return logger

# Initiating a logger
logger = create_logger('model_flow')

# Loading the data from UCI
try:
	logger.info('Connecting to UCI')

	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv')

	logger.info('The data is loaded.')
except:
	logger.critical('The data is not available.', exc_info=True)

# Data preparation
logger.info('Data preparation has started.')

y = df.pop('Air temperature [K]')
X = df[['Process temperature [K]', 'HDF']]
del df

logger.info('Train-test split has started.')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
logger.info('Train-test split is finished.')

logger.info('Data preparation is finished.')

# Model creation
logger.info('Model creation has started.')
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('num', 
                                                numeric_transformer, 
                                                ['Process temperature [K]'])],
                                 remainder='passthrough')
pipe_model = Pipeline([('preprocessor', preprocessor), 
                       ('model', LinearRegression())])
logger.info('Model creation is finished.')

# Model estimation
logger.info('Model estimation has started.')
pipe_model.fit(X_train, y_train)
logger.info('Model estimation is finished.')

def adj_r2(X, y, model):
    r2 = model.score(X, y)
    n = X.shape[0]
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

test_accuracy = adj_r2(X=X_test, y=y_test, model=pipe_model)
logger.info(f"Test accuracy is {test_accuracy:.4}.")

# Saving the model
logger.info('Model saving process has started.')
file = 'regression_model.pickle'
pickle.dump(pipe_model, open(file, 'wb'))
logger.info('Model saving process is finished.')
