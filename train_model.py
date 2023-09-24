from keras.callbacks import ModelCheckpoint, EarlyStopping
from SpatioTempralModel import loadModel
import numpy as np

X_train = np.load('training.npy')
frames = X_train.shape[2]
frames = frames - frames % 10

X_train = X_train[:, :, :frames]
X_train = X_train.reshape(-1, 227, 227, 10)
X_train = np.expand_dims(X_train, axis=4)
Y_train = X_train.copy()

epochs = 1
batch_size = 32     
model = loadModel()

model.compile(optimizer='adam', loss='mean_squared_error')

callback_save = ModelCheckpoint("model.h5", monitor="mean_squared_error", save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

print('Model has been loaded')

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[callback_save, callback_early_stopping]
          )
model.save(filepath=r"C:\Users\HABIB\Documents\FYP\Anomaly-Detection-in-CCTV-Surveillance-Videos-master\STAutoEncoder\model.h5")

