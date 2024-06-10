class TextAIWrapper:

    def __init__(self, model_version="latest", file_path="files"):
        raise NotImplementedError

    def predict(self, X, features_names=None):
        raise NotImplementedError