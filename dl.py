from ludwig.api import LudwigModel
import pandas as pd

training_df = pd.read_csv('transactions.csv')

ludwig_model = LudwigModel(model_definition_file='model_definition.yaml')
train_status = ludwig_model.train(data_df=training_df)
print(max(train_status['validation']['combined']['accuracy']))