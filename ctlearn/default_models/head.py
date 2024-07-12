import tensorflow as tf
import tensorflow.keras.layers as tf_layers
from ctlearn.default_models.basic import fully_connect


def standard_head(inputs, tasks, params):
    # Get the settings for the standard head
    standard_head_settings = params["standard_head"]

    logits = {}
    losses = {}
    loss_weights = {}
    metrics = {}
    if "type" in tasks:
        logit = fully_connect(
            inputs,
            standard_head_settings["type"]["fc_head"],
            expected_logits_dimension=params["num_classes"],
            name="particletype",
        )
        logits["type"] = tf_layers.Softmax(name="type")(logit)
        losses["type"] = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        loss_weights["type"] = standard_head_settings["type"]["weight"]
        metrics["type"] = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    if "energy" in tasks:
        logits["energy"] = fully_connect(
            inputs,
            standard_head_settings["energy"]["fc_head"],
            expected_logits_dimension=1,
            name="energy",
        )
        losses["energy"] = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        loss_weights["energy"] = standard_head_settings["energy"]["weight"]
        metrics["energy"] = tf.keras.metrics.MeanAbsoluteError(name="mae_energy")
    if "direction" in tasks:
        logits["direction"] = fully_connect(
            inputs,
            standard_head_settings["direction"]["fc_head"],
            expected_logits_dimension=3,
            name="direction",
        )
        losses["direction"] = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        loss_weights["direction"] = standard_head_settings["direction"]["weight"]
        metrics["direction"] = tf.keras.metrics.MeanAbsoluteError(name="mae_direction")
    if "cherenkov_photons" in tasks:
        logits["cherenkov_photons"] = fully_connect(
            inputs,
            standard_head_settings["cherenkov_photons"]["fc_head"],
            expected_logits_dimension=1,
            name="cherenkov_photons",
        )
        losses["cherenkov_photons"] = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        loss_weights["cherenkov_photons"] = standard_head_settings["cherenkov_photons"]["weight"]
        metrics["cherenkov_photons"] = tf.keras.metrics.MeanAbsoluteError(name="mae_cherenkov_photons")

    # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
    # https://github.com/keras-team/keras/issues/11735
    if len(tasks) == 1:
        logits = logits[tasks[0]]
        losses = losses[tasks[0]]
        loss_weights = loss_weights[tasks[0]]
        metrics = metrics[tasks[0]]

    return logits, losses, loss_weights, metrics
