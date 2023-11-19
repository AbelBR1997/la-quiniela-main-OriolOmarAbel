import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class QuinielaModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def train(self, train_data):
        train_data = train_data.drop(['home_team', 'away_team'], axis=1)
        # Split the data into features and target
        X = train_data.drop('outcome', axis=1)
        y = train_data['outcome']
 
        y_integers = self.label_encoder.fit_transform(y)
        y_encoded = to_categorical(y_integers, num_classes=3)
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Reshape input data to be 3D [samples, time_steps, features] for RNN
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential()
        model.add(SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(3, activation='softmax')) # Output layer with 3 units for each class

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        self.model = model

    def predict(self, predict_data):
        predict_data = predict_data.drop(['home_team', 'away_team'], axis=1)

        predict_data_scaled = self.scaler.transform(predict_data)
        predict_data_scaled = predict_data_scaled.reshape((predict_data_scaled.shape[0], 1, predict_data_scaled.shape[1]))

        predictions = self.model.predict(predict_data_scaled)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
