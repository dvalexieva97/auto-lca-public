MODEL_PATH = "path_to_model"  # TODO CONFIG


class PaperProcessor:
    summarization_model = None

    def __init__(self) -> None:
        self.load_model()

    @classmethod
    def load_model(cls):
        if not cls.summarization_model:
            cls.summarization_model = cls.get(MODEL_PATH)

    @classmethod
    def get(cls, MODEL_PATH):
        pass

    def summarize():
        pass
