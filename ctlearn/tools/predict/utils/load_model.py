def load_model(self):
    if self.framework_type == "keras":
        from ctlearn.tools.predict.keras.predic_LST1_keras import load_keras_model
        return load_keras_model(self)
    else:
        self.log.error("Framework not found !!!")
        return None