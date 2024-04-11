class SarcasmClassifier:
    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def from_hf(cls, task, name):
        from transformers import pipeline
        return cls(pipeline(task, model=name))
        
    def __preprocess(self, text):
        new_text = []
        for t in text.split(' '):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return ' '.join(new_text)

    def predict(self, text):
        preprocessed_text = self.__preprocess(text)
        return self.pipe(preprocessed_text)[0]['label'] == "LABEL_1"