import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras import regularizers
from functools import lru_cache


@lru_cache
class Model:
    """
    A class with all funcionalities for our model
    """
    def __init__(self):
        self.train_dataset_path = "./datasets/train.csv"
        self.test_dataset_path = "./datasets/test.csv"

        self.dataset = pd.read_csv(self.train_dataset_path)

    def filter_data(self):
        self.data_inputs = self.dataset[["Pclass", "Sex", "Age", "SibSp", "Parch"]].copy()
        self.data_inputs.Age.fillna(self.data_inputs.Age.mean(), inplace=True)
        self.data_inputs.Sex = self.data_inputs.Sex.apply(lambda x: 0 if x == "male" else 1)  # Converting gender in binary expression
        self.data_inputs = self.data_inputs.astype(np.float64)
        self.data_outputs = self.dataset.Survived.to_numpy().astype(np.float64)
    
    def setting_model(self):

        self.classifier = Sequential()
        self.classifier.add(Dense(units=32, activation="elu", kernel_initializer="normal", input_dim=5))
        self.classifier.add(Dropout(0.15))
        self.classifier.add(Dense(units=64, activation="elu", kernel_initializer="normal", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        self.classifier.add(Dropout(0.3))
        self.classifier.add(Dense(units=1, activation="sigmoid"))

    def fitting_model(self):
        self.classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy", "accuracy"])
        self.classifier.fit(self.data_inputs, self.data_outputs, batch_size=200, epochs=4000)

    def testing_model(self):
        self.test_dataset = pd.read_csv(self.test_dataset_path)
        self.classifier.predict(self.test_dataset)



model = Model()
model.filter_data()


