import torch
from transformers import BertTokenizer
import joblib
from app.model import BertForMultiLabelClassification
from app.preprocess import clean_text

class TextClassifierPipeline:
    def __init__(self, model_path: str, tokenizer_path: str, mlb_path: str, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMultiLabelClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.mlb = joblib.load(mlb_path)

    def predict(self, text: str, threshold: float = 0.5):
        """
        Формирует предсказание меток для переданного текста

        Parameters
        ----------
        text : str
            Текст, для которого необходимо предсказать метки.
        threshold : float, optional
            Порог, используемый для классификации, по умолчанию 0.5

        Returns
        -------
        list
            Список предсказанных меток.
        """
        text = clean_text(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs > threshold).astype(int)
        labels = self.mlb.inverse_transform(preds)

        return labels[0]