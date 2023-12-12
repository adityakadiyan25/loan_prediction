test_flag_path = '/Users/adityakadiyan/Downloads/senior_ds_test/data/test/test_flag.csv'
accounts_data_path = '/Users/adityakadiyan/Downloads/senior_ds_test/data/test/accounts_data_test.json'
enquiry_data_path = '/Users/adityakadiyan/Downloads/senior_ds_test/data/test/enquiry_data_test.json'
credit_column_path = 'training_columns.csv'

final_output_path = 'predictions.csv'
model_path = 'my_xgboost_model.json'
features_path ='model_feature_names.txt'
loan_data_features = {
    'loan_amount': ['sum', 'median','mean', 'max', 'min', 'count'],
}
amount_overdue_features = {
    'amount_overdue': ['sum','median', 'mean', 'max', (lambda x: (x > 0).sum()), (lambda x: (x > 0).mean())]
}
date_features = {
    'is_active': 'sum',
    'opened_on_holiday': 'sum',
    'closed_on_holiday': 'sum',
    'loan_duration': ['mean', 'sum', 'max'],
    'time_since_opened': ['mean', 'max'],
    'open_month': lambda x: x.mode()[0],
    'open_quarter': lambda x: x.mode()[0],
    'open_year': lambda x: x.mode()[0],
    'opened_on_weekend': 'sum'
}
enquiry_amt_features = {
    'enquiry_amt': ['sum', 'median','mean', 'max', 'min', 'count'],
}
enquiry_date_features = {
    'days_since_enquiry_time': ['mean', 'max'],
    'enquiry_month': lambda x: x.mode()[0],
    'enquiry_quarter': lambda x: x.mode()[0],
    'enquiry_year': lambda x: x.mode()[0],
    'enquiry_on_weekend': 'sum'
}
payment_hist_features = {
    'recent_delinquency': ['mean','median', 'max', 'min'],
    'max_delinquency': ['mean','median', 'max', 'min'],
    'avg_delinquency': ['mean','median'],
    'late_payments': ['sum','median', 'mean'],
    'consecutive_ontime_payments': ['mean','median', 'max', 'min'],
    'total_payments': ['sum', 'median','mean'],
    'zero_delinquency_months': ['sum', 'median','mean'],
    'late_payment_proportion': ['mean','median'],
    'delinquency_std_dev': ['mean','median']
}