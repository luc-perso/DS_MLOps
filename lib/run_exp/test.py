import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics


@tf.autograph.experimental.do_not_convert
def test_accuracy(y_test_pd, y_pred_pd):
  accuracy = metrics.accuracy_score(y_test_pd, y_pred_pd)
  
  return accuracy


def test_prediction(model, ds_test, batch_size):
  y_pred = model.predict(ds_test.batch(batch_size), batch_size=batch_size)
  y_test = ds_test.map(lambda img, label: label)
  y_test = np.stack(list(y_test))

  y_pred_pd = pd.DataFrame(y_pred, columns=[0, 1, 2]).idxmax(1)
  y_test_pd = pd.DataFrame(y_test, columns=[0, 1, 2]).idxmax(1)

  return y_test_pd, y_pred_pd


def test_conf_mat(y_test_pd, y_pred_pd):
  conf_mat = pd.crosstab(y_test_pd, y_pred_pd, 
                          colnames=['Predicted'],
                          rownames=['Real'],
                          )

  return conf_mat


@tf.autograph.experimental.do_not_convert
def test_report(y_test_pd, y_pred_pd):
  report = metrics.classification_report(y_test_pd, y_pred_pd)

  return report


@tf.autograph.experimental.do_not_convert
def test_model(model, ds_test, batch_size):
  y_test_pd, y_pred_pd = test_prediction(model, ds_test, batch_size)

  accuracy = test_accuracy(y_test_pd, y_pred_pd)
  conf_mat = test_conf_mat(y_test_pd, y_pred_pd)
  report = test_report(y_test_pd, y_pred_pd)
  
  return y_test_pd, y_pred_pd, accuracy, conf_mat, report


@tf.autograph.experimental.do_not_convert
def compile_test_model(model, ds_test, batch_size, from_logits=False, label_smoothing=0.1):
  model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
  )
  y_test_pd, y_pred_pd, accuracy, conf_mat, report = test_model(model, ds_test, batch_size)

  return y_test_pd, y_pred_pd, accuracy, conf_mat, report