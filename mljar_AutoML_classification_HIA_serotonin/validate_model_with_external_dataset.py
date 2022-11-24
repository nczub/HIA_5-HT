from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised.automl import AutoML
from configparser import ConfigParser
from supervised.utils.additional_metrics import AdditionalMetrics

# Read config.ini file
config = ConfigParser(allow_no_value=True)
config.read('test_config.ini')
# read config data
my_automl_directory = config['USERCONFIG']['automl_directory']
my_test_dataset = config['USERCONFIG']['test_dataset']
my_test_dataset_separator = config['USERCONFIG']['test_dataset_separator']
my_test_result_file = config['USERCONFIG']['test_result_file']
buffer = config['USERCONFIG']['header']
my_header = None if buffer is None else int(eval(buffer))
if my_test_dataset != '':
    my_test_data = pd.read_csv(my_test_dataset, sep=my_test_dataset_separator, engine='python', header=my_header)

    ncols = my_test_data.shape[1] - 1

    my_test_data_X = my_test_data.iloc[:, 1:ncols]
    my_test_data_Y = my_test_data.iloc[:, ncols]

    print('my_test_data_X')
    print(my_test_data_X)

    automl = AutoML(results_path=my_automl_directory)
    print('automl')
    print(automl)

    my_ml_task = automl.ml_task
    print('my_ml_task=', my_ml_task)
    # best_model_name = automl.best_model
    predictions = automl.predict_all(my_test_data_X)

    print('predictions')
    print(predictions)
    print('my_test_data_Y')
    print(my_test_data_Y)

    validation_metrics = AdditionalMetrics.compute(target=my_test_data_Y, predictions=predictions, sample_weight=None,
                                                   ml_task=my_ml_task)

    print("validation metrics:", validation_metrics)
    validation_score = automl.score(my_test_data_X, my_test_data_Y)
    print("validation score reg-R2/class-accuracy:", validation_score)
    # print(predictions)
    if predictions.shape[1] > 1:
        df = pd.DataFrame(predictions)
    else:
        # list_of_tuples = list(zip(predictions, my_test_data_Y))
        list_of_tuples = [predictions, my_test_data_Y]
        df = pd.concat(list_of_tuples, axis=1)
        df.columns = ['prediction', 'target']
        # columns = ['prediction', 'target']

    print('df')
    print(df)

    report_file = 't-res_opened_model_' + my_automl_directory + '_external_validation.csv'
    df.to_csv(report_file, index=False)
    if validation_metrics is not None:
        validation_metrics.to_csv(report_file, index=False, mode='a')

    ###### old code
    # validation_RMSE = sqrt(mean_squared_error(my_test_data_Y, predictions))
    # print("validation RMSE:", validation_RMSE)
    # validation_score = automl.score(my_test_data_X, my_test_data_Y)
    # print("validation score reg-R2/class-accuracy:", validation_score)
    # print(predictions)
    # list_of_tuples = list(zip(my_test_data_Y, predictions))
    # df = pd.DataFrame(list_of_tuples, columns=['OBS', 'PRED'])
    # df.to_csv(my_test_result_file, index=False)
    #
    # file = my_test_result_file
    # with open(file, 'a') as f:
    #     f.writelines("\n")
    #     f.writelines(f"RMSE:,{str(validation_RMSE)}" + "\n")
    #     f.writelines(f"SCORE,{str(validation_score)}" + "\n")
#
