# mljar implementation with customized k-fold cv and automated testing with the external the dataset
# Authors: Jakub Szlek, PhD, e-mail: j.szlek@uj.edu.pl, Aleksander Mendyk, PhD, DSc e-mail: aleksander.mendyk@uj.edu.pl
# USE IT AT YOUR OWN RISK!
# License: GPL 3.0
# v9.2 see changelog.txt

import json
import os, fnmatch
import random
import shutil
from datetime import datetime
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    log_loss
)
from supervised.utils.metric import logloss, rmse, negative_f1, negative_accuracy, negative_spearman, negative_pearson

from sklearn.model_selection import GroupKFold
from supervised.automl import AutoML
from supervised.utils.additional_metrics import AdditionalMetrics
from configparser import ConfigParser

# aetting up lists for databse reduction
general_list_orig_cols = []
general_list_additional_cols = []


# computing goodness-of-fit
def goodness_of_fit(name, OBS, PRED, gdf_sample_weight, gdf_labels):
    result = 0
    # try:
    if name == "logloss":
        result = log_loss(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight, labels=gdf_labels)
    elif name == "auc":
        result = roc_auc_score(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight, labels=gdf_labels)
    elif name == "accuracy":
        result = accuracy_score(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight)
    elif name == "rmse":
        result = rmse(y_true=OBS, y_predicted=PRED, sample_weight=gdf_sample_weight)
    elif name == "mse":
        result = mean_squared_error(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight)
    elif name == "mae":
        result = mean_absolute_error(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight)
    elif name == "r2":
        result = r2_score(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight)
    elif name == "mape":
        result = mean_absolute_percentage_error(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight)
    elif name == "spearman":
        result = -negative_spearman(y_true=OBS, y_predicted=PRED, sample_weight=gdf_sample_weight)
    elif name == "pearson":
        result = -negative_pearson(y_true=OBS, y_predicted=PRED, sample_weight=gdf_sample_weight)
    elif name == "f1":
        result = f1_score(y_true=OBS, y_pred=PRED, sample_weight=gdf_sample_weight, average='micro')
    # except:
    # result=999
    return result


# diagnostics for models goodness of fit
def prepare_reduced_database(orig_database, list_orig_cols, list_add_cols, json_object):
    def multiply_columns(col1, col2, operation):
        if operation == "diff":
            return orig_database[col1] - orig_database[col2]
        elif operation == "multiply":
            return orig_database[col1] * orig_database[col2]
        elif operation == "ratio":
            return orig_database[col1] / orig_database[col2]
        elif operation == "sum":
            return orig_database[col1] + orig_database[col2]
        else:
            return None

    # stage 1 based on the original data
    reduced_database = orig_database[list_orig_cols]
    # stage2 create additional data
    if (len(list_add_cols) > 0) and (json_object is not None):
        json_new_columns = json_object.get("new_columns")
        json_new_features = json_object.get("new_features")
        for i in range(len(json_new_columns)):
            # print(json_new_columns[i])
            # print("feature1",json_new_features[i]['feature1'])
            if json_new_columns[i] in list_add_cols:
                buffer_column = multiply_columns(json_new_features[i]['feature1'], json_new_features[i]['feature2'],
                                                 json_new_features[i]['operation'])
                reduced_database.insert(loc=len(reduced_database.columns) - 1, column=json_new_columns[i],
                                        value=buffer_column)
    new_reduced_database = reduced_database.copy()
    return new_reduced_database


def calculate_current_metric(ccm_report_file, ccm_predictions, ccm_my_data_Y, ccm_my_eval_metric, ccm_sample_weight,
                             ccm_labels, forced_result=None):
    result = 0
    list_of_tuples = list(zip(ccm_predictions, ccm_my_data_Y))
    df_0 = pd.DataFrame(list_of_tuples, columns=['prediction', 'target'])
    df_0.to_csv(ccm_report_file, index=False)
    if forced_result is None:
        result = goodness_of_fit(name=ccm_my_eval_metric, OBS=ccm_my_data_Y, PRED=ccm_predictions,
                                 gdf_sample_weight=ccm_sample_weight,
                                 gdf_labels=ccm_labels)
    else:
        result = forced_result
    print("Metric ", ccm_my_eval_metric, " = ", result)

    with open(ccm_report_file, 'a') as f:
        f.writelines("\n")
        f.writelines("Metric " + ccm_my_eval_metric + " = ," + str(result) + "\n")
        f.writelines("\n")
    return result


# provide additional metrics
def calculate_additional_metric(cam_report_file, cam_predictions, cam_my_data_Y, cam_sample_weight):
    result = 0
    try:
        result = AdditionalMetrics.compute(target=cam_my_data_Y, predictions=cam_predictions,
                                           sample_weight=cam_sample_weight, ml_task=my_ml_task)
        print(result["max_metrics"])
        result["max_metrics"].to_csv(cam_report_file, index=False, mode='a')
    except:
        result = None
        print("Could not calculate additional metrics")
    finally:
        shutil.copy(report_file, '.')

    return result


# searching for files
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        result.extend(
            os.path.join(root, name)
            for name in files
            if fnmatch.fnmatch(name, pattern)
        )
    return result


# calculating goodness-of-fit

########## set config based params ###########################################
config = ConfigParser(allow_no_value=True)
config.read('config.ini')

############### user-defined parameters ######################################
# set datasets
train_dataset = config['USERCONFIG']['train_dataset']
# mljar_train_IVIVC_multiple_C.txt'  # train dataset with first column as indices for k-fold cv
test_dataset = config['USERCONFIG']['test_dataset']
# 'mljar_test_IVIVC_multiple_C.txt'  # leave empty to omit this step
dataset_separator = config['USERCONFIG']['dataset_separator']
# '\t'
buffer = config['USERCONFIG']['my_header']
if buffer is None:
    my_header = None
else:
    my_header = int(eval(buffer))
# other settings
index_column = int(config['USERCONFIG']['index_column'])
# 1  # usually first column is the index for governing k-fold cross-validation
no_folds = int(config['USERCONFIG']['no_folds'])
# 10  # k value  k-fold cv
my_total_time_limit = int(eval(config['USERCONFIG']['my_total_time_limit']))
# 2 * 60  # total execution time
buffer = config['USERCONFIG']['my_model_time_limit']
if buffer is None:
    my_model_time_limit = None
else:
    my_model_time_limit = int(eval(buffer))
# None  # per model execution time limit - if None then my_total_time_limit is in force
my_mode = config['USERCONFIG']['my_mode']
# "Compete"  # modes: Explain, Perform, Compete, Optuna
my_ml_task = config['USERCONFIG']['my_ml_task']
# 'regression'  # could be "auto", "binary_classification", "multiclass_classification", "regression"
my_algorithms = config['USERCONFIG']['my_algorithms']
if my_algorithms != 'auto':
    my_algorithms = list(my_algorithms.split(","))
# ['Baseline','Linear','Decision Tree','Random Forest','Extra Trees','LightGBM',
# 'Xgboost','CatBoost','Neural Network','Nearest Neighbors']
print("My algorithms", my_algorithms)
my_train_ensemble = config['USERCONFIG'].getboolean('my_train_ensemble')
# True  # True/False
buffer = config['USERCONFIG']['my_stack_models']
if buffer == 'auto':
    my_stack_models = buffer
else:
    my_stack_models = config['USERCONFIG'].getboolean('my_stack_models')
# 'auto'  # 'auto', True/False
my_eval_metric = config['USERCONFIG']['my_eval_metric']
# 'auto'
# for binary classification: logloss, auc, f1, average_precision, accuracy
# for mutliclass classification: logloss, f1, accuracy - default is logloss (if left "auto")
# for regression: rmse, mse, mae, r2, mape, spearman, pearson - default is rmse (if left "auto")


buffer = config['USERCONFIG']['my_explain_level']
if buffer == 'auto':
    my_explain_level = buffer
else:
    my_explain_level = int(config['USERCONFIG']['my_explain_level'])
# my_explain_level = 'auto' or 0,1,2

buffer = config['USERCONFIG']['my_features_selection']
if buffer == 'auto':
    my_features_selection = buffer
else:
    my_features_selection = config['USERCONFIG'].getboolean('my_features_selection')

# my_features_selection = 'auto'  # 'auto', True/False

buffer = config['USERCONFIG']['my_golden_features']
if buffer == 'auto':
    my_golden_features = buffer
elif (buffer == 'True') or (buffer == 'False'):
    my_golden_features = config['USERCONFIG'].getboolean('my_golden_features')
else:
    my_golden_features = int(config['USERCONFIG']['my_golden_features'])
# 'auto'  # 'auto', True/False
# my_golden_features = 'auto'  # 'auto' True/False or integer to set the number explicitly

buffer = config['USERCONFIG']['my_start_random_models']
if buffer == 'auto':
    my_start_random_models = buffer
else:
    my_start_random_models = int(config['USERCONFIG']['my_start_random_models'])
# my_start_random_models = 'auto'  # 'auto' or int - standard max is 10

buffer = config['USERCONFIG']['my_hill_climbing_steps']
if buffer == 'auto':
    my_hill_climbing_steps = buffer
else:
    my_hill_climbing_steps = int(config['USERCONFIG']['my_hill_climbing_steps'])
# my_hill_climbing_steps = 'auto'  # 'auto' or integer - standard max is 3

buffer = config['USERCONFIG']['my_top_models_to_improve']
if buffer == 'auto':
    my_top_models_to_improve = buffer
else:
    my_top_models_to_improve = int(config['USERCONFIG']['my_top_models_to_improve'])
# my_top_models_to_improve = 'auto'  # 'auto' or integer - standard max is 3

buffer = config['USERCONFIG']['my_boost_on_errors']
if buffer == 'auto':
    my_boost_on_errors = buffer
else:
    my_boost_on_errors = config['USERCONFIG'].getboolean('my_boost_on_errors')
# my_boost_on_errors = 'auto'  # 'auto' or True/False

buffer = config['USERCONFIG']['my_kmeans_features']
if buffer == 'auto':
    my_kmeans_features = buffer
else:
    my_kmeans_features = config['USERCONFIG'].getboolean('my_kmeans_features')
# my_kmeans_features = 'auto'  # 'auto' or True/False

buffer = config['USERCONFIG']['my_mix_encoding']
if buffer == 'auto':
    my_mix_encoding = buffer
else:
    my_mix_encoding = config['USERCONFIG'].getboolean('my_mix_encoding')
# my_mix_encoding = 'auto'  # 'auto or True/False

my_max_single_prediction_time = config['USERCONFIG']['my_max_single_prediction_time']
if my_max_single_prediction_time is not None:
    my_max_single_prediction_time = int(eval(config['USERCONFIG']['my_max_single_prediction_time']))
# None  # time in seconds
my_optuna_time_budget = config['USERCONFIG']['my_optuna_time_budget']
if my_optuna_time_budget is not None:
    my_optuna_time_budget = int(eval(config['USERCONFIG']['my_optuna_time_budget']))
# None  # time in seconds
my_optuna_verbose = config['USERCONFIG'].getboolean('my_optuna_verbose')
# True/False
my_n_jobs = int(config['USERCONFIG']['my_n_jobs'])
# -1  # integer
my_random_state = config['USERCONFIG']['my_random_state']
if my_random_state is not None:
    my_random_state = int(config['USERCONFIG']['my_random_state'])
# None - means total randomness otherwise set an integer

############### main code ######################################
# random state
if my_random_state is None:
    my_random_state = random.randint(0, 1000000)
    print(f"Random seed: {str(my_random_state)}")

# Load the data
my_data = pd.read_csv(train_dataset, sep=dataset_separator, engine='python', header=my_header)

# read current date and time
now = datetime.now()  # current date and time

# set AutoML_directory
AutoML_directory = f"mljar_AutoML_{my_mode}_" + now.strftime(
    "%Y_%m_%d_%H_%M_%S"
)

ncols = my_data.shape[1] - 1
nrows = my_data.shape[0]

# needed to make cv by groups - first column contains indices!
groups_index = my_data.columns[index_column - 1]
# group rows by group label
grp = [my_data for _, my_data in my_data.groupby(groups_index)]
# random shuffle groups with seed current_my_random_seed_10cv
random.Random().shuffle(grp)
# concat the groups
my_data = pd.concat(grp).reset_index(drop=True)

# set training data and labels
my_data_X = my_data.drop(my_data.columns[[0, ncols]], axis=1)
my_data_Y = my_data[my_data.columns[ncols]]

# needed to make cv by groups - first column contains indices!
groups = my_data[my_data.columns[[index_column - 1]]]

# create GroupKFold
sgkf = GroupKFold(n_splits=no_folds).split(my_data_X, my_data_Y, groups=groups)
# gsk = GroupShuffleSplit(n_splits=no_folds, random_state=current_my_random_seed_10cv)


# train models with AutoML
automl = AutoML(mode=my_mode,
                validation_strategy={"validation_type": "custom"},
                total_time_limit=my_total_time_limit,
                model_time_limit=my_model_time_limit,
                ml_task=my_ml_task,
                results_path=AutoML_directory,
                algorithms=my_algorithms,
                train_ensemble=my_train_ensemble,
                stack_models=my_stack_models,
                eval_metric=my_eval_metric,
                explain_level=my_explain_level,
                features_selection=my_features_selection,
                golden_features=my_golden_features,
                start_random_models=my_start_random_models,
                hill_climbing_steps=my_hill_climbing_steps,
                top_models_to_improve=my_top_models_to_improve,
                boost_on_errors=my_boost_on_errors,
                kmeans_features=my_kmeans_features,
                mix_encoding=my_mix_encoding,
                max_single_prediction_time=my_max_single_prediction_time,
                optuna_time_budget=my_optuna_time_budget,
                optuna_verbose=my_optuna_verbose,
                n_jobs=my_n_jobs,
                random_state=my_random_state)
automl.fit(my_data_X, my_data_Y, cv=list(sgkf))

# reporting and identifying best model
print("GENERATING REPORT")
automl.report()
automl.select_and_save_best()
my_params = automl.get_params()
print("my_params = ", my_params)
best_model_name = automl._best_model.get_name()
best_model_cv_metric = automl._best_model.get_final_loss()
print("Best model name:", best_model_name)
print("Best model metric: = ", best_model_cv_metric)
print("automl.eval_metric:", automl.eval_metric)
fname = AutoML_directory + '/params.json'
if os.path.isfile(fname):
    params_from_json = json.load(open(fname, "r"))
    my_eval_metric_from_json = params_from_json.get("eval_metric")
    print("Eval metric is ", my_eval_metric_from_json)
print("")

# get training diagnostics
print("TRAINING")
predictions = automl.predict(my_data_X)

# print("predictions",predictions)
# print("targets",my_data_Y)

report_file = AutoML_directory + '/t-res_best_model_' + AutoML_directory + '_' + best_model_name + '_training.csv'

train_metrics = calculate_current_metric(ccm_report_file=report_file, ccm_predictions=predictions,
                                         ccm_my_data_Y=my_data_Y,
                                         ccm_my_eval_metric=my_eval_metric_from_json, ccm_sample_weight=None,
                                         ccm_labels=None)

additonal_train_metrics = calculate_additional_metric(cam_report_file=report_file, cam_predictions=predictions,
                                                      cam_my_data_Y=my_data_Y, cam_sample_weight=None)

train_score = automl.score(my_data_X, my_data_Y)
print("Training score reg-R2/class-accuracy:", train_score)

# getting k-fold-cv metrics
print("K-FOLD-CV")
results_files = find('predictions_*.csv', AutoML_directory + "/" + best_model_name)
print("Results files ", results_files[0])

my_cv_predictions = pd.read_csv(results_files[0])
my_cv_target = my_cv_predictions['target']
my_cv_pred_buffer = my_cv_predictions.drop('target', axis=1)
# print("my_cv_predictions",my_cv_predictions)
report_file = AutoML_directory + '/t-res_best_model_' + AutoML_directory + '_' + best_model_name + '_k_fold_cv.csv'

my_cv1 = calculate_current_metric(ccm_report_file=report_file, ccm_predictions=my_cv_pred_buffer,
                                  ccm_my_data_Y=my_cv_target,
                                  ccm_my_eval_metric=my_eval_metric_from_json, ccm_sample_weight=None, ccm_labels=None,
                                  forced_result=abs(best_model_cv_metric))

additional_best_model_cv_metric = calculate_additional_metric(cam_report_file=report_file,
                                                              cam_predictions=my_cv_pred_buffer,
                                                              cam_my_data_Y=my_cv_target, cam_sample_weight=None)

validation_metrics = 0
validation_score = 0

# external validation
print("VALIDATION")
if test_dataset is not None:
    my_test_data = pd.read_csv(test_dataset, sep=dataset_separator, engine='python', header=my_header)
    # compute the MSE on test data

    my_test_data_X = my_test_data.iloc[:, 1:ncols]
    my_test_data_Y = my_test_data.iloc[:, ncols]

    predictions = automl.predict(my_test_data_X)

    report_file = AutoML_directory + '/t-res_best_model_' + AutoML_directory + '_' + best_model_name + '_external_validation.csv'

    validation_metrics = calculate_current_metric(ccm_report_file=report_file, ccm_predictions=predictions,
                                                  ccm_my_data_Y=my_test_data_Y,
                                                  ccm_my_eval_metric=my_eval_metric_from_json, ccm_sample_weight=None,
                                                  ccm_labels=None)

    additonal_validation_metrics = calculate_additional_metric(cam_report_file=report_file, cam_predictions=predictions,
                                                               cam_my_data_Y=my_test_data_Y, cam_sample_weight=None)

    validation_score = automl.score(my_test_data_X, my_test_data_Y)
    print("validation score reg-R2/class-accuracy:", validation_score)
    # print(predictions)

# reporting for high throughput analysis
file = AutoML_directory + '/short_out.txt'
line1 = "Train_metric" + "\t" + "Train_score" + "\t" + "k_fold_cv_metric" + "\t" + "External_validation_metric" + "\t" + "External_validation_score" + "\n"
line2 = str(train_metrics) + "\t" + str(train_score) + "\t" + str(
    best_model_cv_metric) + "\t" + str(
    validation_metrics) + "\t" + str(validation_score) + "\n"
with open(file, 'w') as f:
    f.writelines(line2)
shutil.copy(file, '.')

# checking golden_features
if os.path.isfile(AutoML_directory + '/golden_features.json'):
    fname = AutoML_directory + '/golden_features.json'
    retrieved_golden_features_json = json.load(open(fname, "r"))
    retrieved_golden_features_list = retrieved_golden_features_json.get("new_columns")
else:
    retrieved_golden_features_list = None
    retrieved_golden_features_json = None

# finding which features could be dropped
if os.path.isfile(AutoML_directory + '/drop_features.json'):
    # read and print to file columns to be dropped
    cols_to_drop = pd.read_json(AutoML_directory + '/drop_features.json')
    cols_to_drop.to_csv(AutoML_directory + '/cols_to_drop.csv')
    shutil.copy(AutoML_directory + '/cols_to_drop.csv', '.')
    # create a final list for determination of which columns should be dropped
    cols_to_drop_list = cols_to_drop.values
    # concatenate origianl columns and these created as golden features (if any)
    if retrieved_golden_features_list is not None:
        buffer_list = list(my_data.columns) + retrieved_golden_features_list
    else:
        buffer_list = list(my_data.columns)
    # find the columns after dropping the unnecessary ones
    left_names = []
    for item in buffer_list:
        if item not in cols_to_drop_list:
            left_names = left_names + [item]
    # indices from original database
    rows_left_names = []
    for col_select in left_names:
        try:
            # indices in a natural scale
            index_no = my_data.columns.get_loc(col_select) + 1
            general_list_orig_cols = general_list_orig_cols + [col_select]
        except:
            index_no = -999
            general_list_additional_cols = general_list_additional_cols + [col_select]
        # print("Index ", index_no, " ", col_select)
        rows_left_names.append([index_no, col_select])

    cols_to_use = pd.DataFrame(rows_left_names, columns=["Index", "Label"])
    cols_to_use = cols_to_use.sort_values(by='Index')
    cols_to_use.to_csv(AutoML_directory + '/cols_to_use.csv', index=False)
    shutil.copy(AutoML_directory + '/cols_to_use.csv', '.')
    new_database = prepare_reduced_database(my_data, general_list_orig_cols, general_list_additional_cols,
                                            retrieved_golden_features_json)
    new_database.to_csv(AutoML_directory + '/reduced_database.csv', index=False)
    shutil.copy(AutoML_directory + '/reduced_database.csv', '.')

# preparing config for testing model
shutil.copy("source_test_config.txt", "test_config.ini")
file = 'test_config.ini'
with open(file, 'a') as f:
    f.writelines("automl_directory = " + AutoML_directory + "\n")
