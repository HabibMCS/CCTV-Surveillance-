from keras.models import load_model
import numpy as np

def mean_squared(x1, x2):
    difference = x1 - x2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_diff = difference ** 2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    mean_dist = dist / n_samples
    return mean_dist

threshold = 0.0005

model = load_model('model.h5')

X_test = np.load('testing.npy')  # Use testing.npy for testing, assuming it contains the test data
frames = X_test.shape[2]

# Calculate the number of frames to pad to reach a multiple of 10
frames_to_pad = (10 - (frames % 10)) % 10

# Pad the frames with duplicates (you can also use zero padding if desired)
X_test = np.pad(X_test, ((0, 0), (0, 0), (0, frames_to_pad)), mode='edge')

# Reshape to create batches of 10 frames
X_test = X_test.reshape(-1, 227, 227, 10)
X_test = np.expand_dims(X_test, axis=4)
print(X_test)

flag = 0  # Overall video flag

for number, bunch in enumerate(X_test):
    # The rest of your code to detect anomalies and print the results
    n_bunch = np.expand_dims(bunch, axis=0)
    reconstructed_bunch = model.predict(n_bunch)

    loss = mean_squared(n_bunch, reconstructed_bunch)
    print(loss)

    if loss > threshold:
        print("Anomalous bunch of frames at bunch number {}".format(number))
        flag = 1
    else:
        print("..")
        print('Bunch Normal')

if flag == 1:
    print("Anomalous Events detected")
