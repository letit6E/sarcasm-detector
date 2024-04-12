class SarcasmClassifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_hf(cls, name):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        return cls(
            AutoModelForSequenceClassification.from_pretrained(name),
            AutoTokenizer.from_pretrained(name)
        )
        
    def __preprocess(self, text):
        new_text = []
        for t in text.split(' '):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return ' '.join(new_text)

    def predict(self, text):
        tokenized_text = self.tokenizer([self.__preprocess(text)], padding=True, truncation=True, max_length=4096, return_tensors="pt")
        output = self.model(**tokenized_text)
        probs = output.logits.softmax(dim=-1).tolist()[0]
        return probs[0] < 0.5