import numpy as np
def get_predictions(steps , train_data, tf_learner, batch_length) -> list[float]:
    predictions = []
    train_batch = train_data[-batch_length:].reshape((1, batch_length, 1))
    for i in range(steps):
        curr = tf_learner.predict(train_batch)[0]
        predictions.append(curr)
        train_batch = np.append(train_batch[:, 1:, :], [[curr]], axis = 1)
    return predictions
