import keras
from sklearn.metrics import root_mean_squared_error


class BaseModel:
    def __init__(self, input_shape, output_shape, y_scaler):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.y_scaler = y_scaler
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError("Subclasses should implement this method!")

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, x_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def evaluate_model(self, x, y):
        predictions = self.model.predict(x)
        predictions_rescaled = self.y_scaler.inverse_transform(predictions.reshape(-1, 1))
        y_rescaled = self.y_scaler.inverse_transform(y.reshape(-1, 1))
        rmse = root_mean_squared_error(y_rescaled, predictions_rescaled)
        return rmse
