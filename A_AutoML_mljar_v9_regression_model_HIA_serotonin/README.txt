mljar.py - major script for AutoML
config.ini - configuration file for the above script
master_config.ini - source config for the above config
validate_model_with_external_dataset.py - a script for validation of the trained model with the additional data
source_test_config.txt a source for config_test.ini used by the above validate_model_with_external_dataset.py

report in the short_out
if features selection is performed then cols_to_drop.csv contains columns to be excluded form the database and cols_to_use.csv contains variables used by the model otgether woth golden features developed by the software
