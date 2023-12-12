import pandas as pd
import json
import os
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from config import *

def load_json_data(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        with open(file_path, 'r') as file:
            # Load JSON data from the file
            data = json.load(file)
            return data
    except json.JSONDecodeError:
        # Handle JSON decode error (e.g., malformed JSON)
        raise ValueError(f"The file {file_path} is not a valid JSON file.")
    except PermissionError:
        # Handle file permission error
        raise PermissionError(f"Permission denied when trying to read {file_path}.")
    except Exception as e:
        # Handle other potential errors
        raise IOError(f"An error occurred while reading {file_path}: {str(e)}")

def convert_to_dataframe(data):

    # Data type validation
    if not isinstance(data, list) or not all(isinstance(inner_list, list) for inner_list in data):
        raise TypeError("Input data must be a list of lists.")

    # Handle empty data
    if not data or not any(data):
        return pd.DataFrame()

    # Flatten the nested list
    flattened_data = []
    for inner_list in data:
        if inner_list:  # Skip empty inner lists
            flattened_data.extend(inner_list)

    # Create DataFrame
    try:
        df = pd.DataFrame(flattened_data)
    except ValueError as ve:
        raise ValueError(f"Error in creating DataFrame: {ve}")

    return df


def extract_features(payment_hist, uid):
    # Data type validation
    if not isinstance(payment_hist, str):
        raise TypeError("payment_hist must be a string.")
    
    # Check if pandas is available
    if 'pd' not in globals():
        raise ImportError("pandas library is not imported.")

    # Convert payment history to integers with error handling
    try:
        payments = [int(payment_hist[i:i+3]) for i in range(0, len(payment_hist), 3)][::-1] if payment_hist else [0]
    except ValueError:
        raise ValueError("payment_hist contains non-numeric values.")

    total_payments = len(payments)
    late_payments = sum(p > 0 for p in payments)

    # Calculate features
    try:
        return {
            'uid': uid,
            'recent_delinquency': payments[0],
            'max_delinquency': max(payments),
            'avg_delinquency': sum(payments) / total_payments if total_payments else 0,
            'late_payments': late_payments,
            'consecutive_ontime_payments': next((i for i, p in enumerate(payments) if p > 0), total_payments),
            'total_payments': total_payments,
            'zero_delinquency_months': sum(p == 0 for p in payments),
            'late_payment_proportion': late_payments / total_payments if total_payments else 0,
            'delinquency_std_dev': pd.Series(payments).std() if total_payments > 1 else 0
        }
    except Exception as e:
        raise Exception(f"Error calculating features: {e}")



def aggregate_data(df, groupby_column, agg_dict):

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Check for the existence of the groupby column
    if groupby_column not in df.columns:
        raise ValueError(f"The groupby column '{groupby_column}' does not exist in the DataFrame.")

    # Check for the existence of the columns specified in agg_dict
    missing_columns = [col for col in agg_dict if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for aggregation: {missing_columns}")

    # Aggregation with error handling
    try:
        agg_data = df.groupby(groupby_column).agg(agg_dict).reset_index()
    except Exception as e:
        raise Exception(f"Error during aggregation: {e}")

    return agg_data


def determine_active_status(accounts_data):
    if 'closed_date' not in accounts_data.columns:
        raise ValueError("Missing 'closed_date' column in DataFrame.")
    
    accounts_data['is_active'] = accounts_data['closed_date'].isna()
    return accounts_data


def handle_dates_and_holidays(accounts_data):
    if not all(col in accounts_data.columns for col in ['open_date', 'closed_date']):
        raise ValueError("Missing 'open_date' or 'closed_date' columns in DataFrame.")

    accounts_data['open_date'] = pd.to_datetime(accounts_data['open_date'])
    accounts_data['closed_date'] = pd.to_datetime(accounts_data['closed_date'])

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=accounts_data['open_date'].min(), end=accounts_data['closed_date'].max())

    accounts_data['opened_on_holiday'] = accounts_data['open_date'].isin(holidays)
    accounts_data['closed_on_holiday'] = accounts_data['closed_date'].isin(holidays)

    return accounts_data

def calculate_loan_duration(accounts_data):
    if not all(col in accounts_data.columns for col in ['open_date', 'is_active']):
        raise ValueError("Missing required columns in DataFrame.")

    today = datetime.today()
    accounts_data['loan_duration'] = accounts_data.apply(
        lambda row: (row['closed_date'] if not row['is_active'] else today) - row['open_date'], axis=1)

    accounts_data['loan_duration'] = accounts_data['loan_duration'].dt.days
    accounts_data = accounts_data[accounts_data['loan_duration'] >= 0]

    return accounts_data

def add_date_related_features(accounts_data):
    if 'open_date' not in accounts_data.columns:
        raise ValueError("Missing 'open_date' column in DataFrame.")

    today = datetime.today()
    accounts_data['time_since_opened'] = (today - accounts_data['open_date']).dt.days

    accounts_data['open_month'] = accounts_data['open_date'].dt.month
    accounts_data['open_quarter'] = accounts_data['open_date'].dt.quarter
    accounts_data['open_year'] = accounts_data['open_date'].dt.year
    accounts_data['opened_on_weekend'] = accounts_data['open_date'].dt.weekday > 4
    return accounts_data


def process_accounts_data(accounts_data):
    accounts_data = determine_active_status(accounts_data)
    accounts_data = handle_dates_and_holidays(accounts_data)
    accounts_data = calculate_loan_duration(accounts_data)
    accounts_data = add_date_related_features(accounts_data)
    return accounts_data


def process_enquiry_data(enquiry_data):
    # Input validation
    if not isinstance(enquiry_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Check for the existence of the enquiry_date column
    if 'enquiry_date' not in enquiry_data.columns:
        raise ValueError("Missing 'enquiry_date' column in DataFrame.")

    try:
        # Convert enquiry_date to datetime
        enquiry_data['enquiry_date'] = pd.to_datetime(enquiry_data['enquiry_date'])

        # Calculate days since enquiry
        today = datetime.today()
        enquiry_data['days_since_enquiry'] = (today - enquiry_data['enquiry_date']).dt.days

        # Extract month, quarter, and year
        enquiry_data['enquiry_month'] = enquiry_data['enquiry_date'].dt.month
        enquiry_data['enquiry_quarter'] = enquiry_data['enquiry_date'].dt.quarter
        enquiry_data['enquiry_year'] = enquiry_data['enquiry_date'].dt.year

        # Determine if the enquiry was made on a weekend
        enquiry_data['enquiry_on_weekend'] = enquiry_data['enquiry_date'].dt.weekday > 4

        # Handle potential negative days (future dates)
        if enquiry_data['days_since_enquiry'].min() < 0:
            raise ValueError("Enquiry data contains future dates.")
    except Exception as e:
        raise Exception(f"Error processing enquiry data: {e}")

    return enquiry_data
