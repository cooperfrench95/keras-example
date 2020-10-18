import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('./model')


VALIDATION_PICKLE_FEATURES = 'testing_mask_features.pickle'
VALIDATION_PICKLE_LABELS = 'testing_mask_labels.pickle'

with open(VALIDATION_PICKLE_FEATURES, 'rb') as f:
    features_validate = pickle.load(f)

with open(VALIDATION_PICKLE_LABELS, 'rb') as f:
    labels_validate = pickle.load(f)


class Callbacks(tf.keras.callbacks.Callback):
    def on_test_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))


# # prediction = model.predict(features_validate, callbacks=[Callbacks()])
# prediction_counts = {}

# for index, i in enumerate(features_validate):
#     testing = features_validate[-1 * (index + 1):]
#     results = model.predict(testing)
#     pred = float(np.argmax(results[0]))
#     label = labels_validate[-1 * (index + 1)]
#     print('prediction', pred, 'actual', label)
#     if not str(pred) in prediction_counts:
#         prediction_counts[str(pred)] = 0
#     prediction_counts[str(pred)] += 1
#     if pred == label:
#         print('Correct')

# print(prediction_counts)


results = model.evaluate(
    features_validate, labels_validate)
print('test loss, test acc:', results)
