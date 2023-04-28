import tensorflow as tf
from ctlearn.default_models.basic import fully_connect


def standard_head(inputs, tasks, params):

    # Get the settings for the standard head
    standard_head_settings = params["standard_head"]

    logits = {}
    losses = {}
    loss_weights = {}
    metrics = {}
    if "particletype" in tasks:
        logit = fully_connect(
            inputs,
            standard_head_settings["particletype"]["fc_head"],
            expected_logits_dimension=params["num_classes"],
            name="particle",
        )
        logits["particletype"] = tf.keras.layers.Softmax(name="particletype")(logit)
        losses["particletype"] = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        loss_weights["particletype"] = standard_head_settings["particletype"]["weight"]
        metrics["particletype"] = [
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
            expected_logits_dimension=2,
            name="direction",
        )
        losses["direction"] = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        loss_weights["direction"] = standard_head_settings["direction"]["weight"]
        metrics["direction"] = tf.keras.metrics.MeanAbsoluteError(name="mae_direction")

    # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
    # https://github.com/keras-team/keras/issues/11735
    if len(tasks) == 1:
        logits = logits[tasks[0]]
        losses = losses[tasks[0]]
        loss_weights = loss_weights[tasks[0]]
        metrics = metrics[tasks[0]]

    return logits, losses, loss_weights, metrics
