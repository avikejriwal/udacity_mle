from autogluon.tabular import TabularPredictor
# or from autogluon.multimodal import MultiModalPredictor for example
from io import BytesIO, StringIO

import os
import pandas as pd
from sklearn import metrics
from autogluon.tabular import TabularPredictor

TARGET_COL = 'propensity'


def test(model, test_data):
    print("Testing model...")
    
    test_y = test_data[TARGET_COL]

    predictions = model.predict(test_data)
    print('f1:', metrics.f1_score(test_y, predictions))
    print('accuracy:', metrics.accuracy_score(test_y, predictions))
    print('precision:', metrics.precision_score(test_y, predictions))
    print('recall:', metrics.recall_score(test_y, predictions))
    

def train(
    train_data
):
    print('Training model...')
    
    model = TabularPredictor(
        label='propensity', eval_metric='f1',
        problem_type='binary',
        path='model/'
    ).fit(train_data, time_limit=300,
          presets='best_quality'
    )
    
    print(model.fit_summary())
    
    return model
    

def load_data():
    data_source = os.environ["SM_CHANNEL_TRAINING"]

    train_source = os.path.join(data_source, "train.csv")
    test_source = os.path.join(data_source, "test.csv")

    train_data = pd.read_csv(train_source, index_col = 0)
    test_data = pd.read_csv(test_source, index_col = 0)
    
    return train_data, test_data


def main():
    train_data, test_data = load_data()
    model = train(train_data)
    test(model, test_data)
    

if __name__ == "__main__":
    main()