import json
import re
from collections import Counter
from pathlib import Path


# clean_text из ноутбука (ячейка 15) 
def clean_text(text: str) -> str:
    """Очистка текста — точная копия из ноутбука."""
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r"[^a-z0-9\s!?.,\'\-]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Vocabulary из ноутбука (ячейка 18)
class Vocabulary:
    """
    Словарь для токенизации текста.
    Точная копия класса из ноутбука — важно сохранять совместимость,
    чтобы индексы слов совпадали с теми, что были при обучении.
    """
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: dict = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word: dict = {0: '<PAD>', 1: '<UNK>'}

    def build(self, texts: list[str]) -> None:
        """Строит словарь из списка текстов (используется в ноутбуке при обучении)."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        """Токенизирует текст → список индексов длиной max_len (с паддингом)."""
        tokens = [self.word2idx.get(w, self.UNK_IDX) for w in text.split()[:max_len]]
        tokens += [self.PAD_IDX] * (max_len - len(tokens))
        return tokens

    def __len__(self) -> int:
        return len(self.word2idx)

    # Сериализация: сохранение и загрузка словаря 

    def save(self, path: str | Path) -> None:
        """Сохраняет word2idx в JSON-файл."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Загружает словарь из JSON-файла (сохранённого методом save)."""
        with open(path, encoding='utf-8') as f:
            word2idx: dict = json.load(f)
        obj = cls()
        obj.word2idx = word2idx
        obj.idx2word = {v: k for k, v in word2idx.items()}
        return obj
