import pandas as pd
import json
import os
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import xgboost as xgb
from joblib import Parallel, delayed
from config import *
from data_preperation_functions import *
import logging
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def full_data_processing_pipeline():
    try:
        logging.info("Starting the data processing pipeline.")

        # Load Test Flag Data
        logging.info("Loading test flag data.")
        test_flag = pd.read_csv(test_flag_path)
        
        # Load Account and Enquiry Data
        logging.info("Loading accounts data.")
        accounts_data_dict = load_json_data(accounts_data_path)
        
        logging.info("Loading enquiry data.")
        enquiry_data_dict = load_json_data(enquiry_data_path)

        # Convert to DataFrame
        logging.info("Converting accounts data to DataFrame.")
        accounts_data = convert_to_dataframe(accounts_data_dict)
        
        logging.info("Converting enquiry data to DataFrame.")
        enquiry_data = convert_to_dataframe(enquiry_data_dict)

        # Dropping Duplicates
        logging.info("Dropping duplicates in the datasets.")
        accounts_data.drop_duplicates(keep='first', inplace=True)
        enquiry_data.drop_duplicates(keep='first', inplace=True)
        test_flag.drop_duplicates(keep='first', inplace=True)

        # Parallel Processing for Features Extraction
        logging.info("Extracting features using parallel processing.")
         # Adjust based on your machine's capabilities
        results = Parallel(n_jobs=-1)(delayed(extract_features)(
            row.payment_hist_string, row.uid) for index, row in accounts_data.iterrows())
        features_df = pd.DataFrame(results)

        # Aggregate Payment Data
        logging.info("Aggregating payment data.")
        agg_payment_data = aggregate_data(features_df, 'uid', payment_hist_features)
        agg_user_data = agg_payment_data


        logging.info("Aggregating account data by credit type.")
        agg_accounts_data = accounts_data.groupby('uid')['credit_type'].value_counts().unstack(fill_value=0)
        agg_accounts_data.columns = [f'credit_type_count_{col}' for col in agg_accounts_data.columns]

        logging.info("Reading training credit columns.")
        training_credit_columns = pd.read_csv(credit_column_path, header=None).squeeze()

        logging.info("Ensuring all training credit columns are present in aggregated account data.")
        for column in training_credit_columns:
            if (column not in agg_accounts_data.columns) & (column != '0'):
                agg_accounts_data[column] = 0

        logging.info("Aggregating loan data.")
        agg_loan_data = aggregate_data(accounts_data, 'uid', loan_data_features)

        logging.info("Merging loan amount data with credit type data.")
        agg_accounts_data = pd.merge(agg_accounts_data, agg_loan_data, on='uid', how='left')

        logging.info("Aggregating overdue data.")
        agg_overdue_data = aggregate_data(accounts_data, 'uid', amount_overdue_features)

        logging.info("Merging overdue data with account data.")
        agg_accounts_data = pd.merge(agg_accounts_data, agg_overdue_data, on='uid', how='left')

        logging.info("Merging user data.")
        agg_user_data = pd.merge(agg_accounts_data, agg_user_data, on='uid', how='left')

        logging.info("Processing accounts data.")
        accounts_data = process_accounts_data(accounts_data)

        logging.info("Aggregating date data.")
        agg_date_data = aggregate_data(accounts_data, 'uid', date_features)

        logging.info("Final merging of user data with date data.")
        agg_user_data = pd.merge(agg_user_data, agg_date_data, on='uid', how='left')

        logging.info("Data aggregation and merging completed successfully.")

        logging.info("Aggregating enquiry type data.")
        agg_enquiry_type_data = enquiry_data.groupby('uid')['enquiry_type'].value_counts().unstack(fill_value=0)
        agg_enquiry_type_data.columns = [f'enquiry_type_count_{col}' for col in agg_enquiry_type_data.columns]

        logging.info("Aggregating enquiry amount data.")
        agg_amt_data = aggregate_data(enquiry_data, 'uid', enquiry_amt_features)

        logging.info("Merging enquiry amount data with enquiry type data.")
        agg_enquiry_type_data = pd.merge(agg_enquiry_type_data, agg_amt_data, on='uid', how='left')

        logging.info("Processing enquiry data.")
        enquiry_data = process_enquiry_data(enquiry_data)

        logging.info("Aggregating enquiry date data.")
        agg_enquiry_date_data = aggregate_data(enquiry_data, 'uid', enquiry_amt_features)

        logging.info("Merging enquiry date data with enquiry type data.")
        agg_enquiry_type_data = pd.merge(agg_enquiry_type_data, agg_enquiry_date_data, on='uid', how='left')

        logging.info("Merging UID history data.")
        agg_uids_history = pd.merge(agg_enquiry_type_data, agg_user_data, on='uid', how='left').fillna(0)

        logging.info("Processing test flag data.")
        test_flag = pd.get_dummies(test_flag, columns=['NAME_CONTRACT_TYPE'])

        logging.info("Merging test flag data with UID history.")
        test_flag = pd.merge(test_flag, agg_uids_history, on='uid', how='left')

        logging.info("Enquiry data processing and merging completed successfully.")
        test_flag.columns = [str(col) for col in test_flag.columns]
        uids_val = list(test_flag['uid'])

        test_flag.columns = test_flag.columns.astype(str).str.replace(
            '[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace(
            '(', '').str.replace(')', '').str.replace('"', '').str.replace("'", '').str.replace(
            ',', '_').str.replace('lambda', '').str.replace(' ', '_')


        logging.info("Loading feature names from file.")
        feature_names = []
        with open(features_path, 'r') as file:
            feature_names = [line.strip() for line in file]

        logging.info("Selecting features for the final test data.")
        test_flag_final = test_flag[feature_names]

        logging.info("Creating a new XGBoost model instance and loading the model.")
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        logging.info("Making predictions with the model.")
        predictions = model.predict_proba(test_flag_final)

        logging.info("Creating a DataFrame for predictions and merging with UIDs.")
        predictions_df = pd.DataFrame(predictions, columns=['0 Probability', '1 Probability'])
        predictions_df['uids'] = uids_val

        logging.info("Saving predictions to CSV file.")
        predictions_df.to_csv(final_output_path, index=False)
        logging.info(f"Predictions successfully saved to {final_output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



if __name__ == "__main__":
    full_data_processing_pipeline()