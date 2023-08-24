import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the model
        Args:
            X_train (pandas df): Training data
            y_train (pandas series): Training labels
        """
        
        pass
    
    
class LinearRegressionModel(Model):
    """
    Trains the model
    """    
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train (pandas df): Training data
            y_train (pandas series): Training labels
        Returns:
            model: LinearRegressionModel
        """
        
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model training completed successfully")
            return reg
        except Exception as e:
            logging.error("Error in Model training: {}".format(e))
            raise e