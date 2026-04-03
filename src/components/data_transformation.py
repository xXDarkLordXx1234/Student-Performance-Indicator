import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer(self):
        try:
            logging.info("Data Transformation Pipeline Started")
            num_col = ["writing_score","reading_score"]
            categ_col = ["gender","race_ethnicity","parental_level_of_education	","lunch","test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            
            categ_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler())
                ]
            )
            logging.info(f"Categorical Columns: {categ_col}")
            logging.info(f"Numerical columns: {num_col}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_col),
                    ("cat_pipeline",categ_pipeline,categ_col)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer()
            target_col = "math_score"
            num_col = ["writing_score","reading_score"]
            input_feature_train_df = train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df = train_df[target_col]
            input_feature_test_df = test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df = test_df[target_col]
            
            logging.info("Preprocessing training and testing dataframes")
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Saved Preprocessing object")
            
            save_object (
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)