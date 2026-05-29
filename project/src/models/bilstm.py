"""
BiLSTM-классификатор эмоций.

Архитектура взята ДОСЛОВНО из ноутбука (ячейка 26):
    Embedding → BiLSTM (2 слоя) → Attention → FC(256→128→6)

6 эмоций (EMOTION_MAP из ячейки 7 ноутбука):
    0: sadness | 1: joy | 2: love | 3: anger | 4: fear | 5: surprise
"""
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.data.preprocessing import Vocabulary, clean_text

logger = logging.getLogger(__name__)

# ── Маппинг меток (ячейка 7 ноутбука) ────────────────────────────────────────
EMOTION_MAP: Dict[int, str] = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
NUM_CLASSES = len(EMOTION_MAP)   # 6
MAX_LEN     = 64                 # max_len из ноутбука


# ── Архитектура (ячейка 26 ноутбука) ─────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Двунаправленный LSTM с механизмом внимания (attention).

    Параметры по умолчанию совпадают с теми, что использовались при обучении:
        embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.4
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 128,
        hidden_dim:  int = 256,
        num_layers:  int = 2,
        num_classes: int = NUM_CLASSES,
        dropout:     float = 0.4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb          = self.dropout(self.embedding(x))           # [B, L, E]
        out, _       = self.lstm(emb)                            # [B, L, 2H]
        attn_weights = torch.softmax(self.attention(out), dim=1) # [B, L, 1]
        context      = (attn_weights * out).sum(dim=1)           # [B, 2H]
        return self.fc(context)                                  # [B, num_classes]


# ── Загрузка модели + инференс ────────────────────────────────────────────────
class EmotionClassifier:
    """
    Обёртка над BiLSTM: загружает словарь и веса, делает предсказания.

    Параметры
    ---------
    vocab_path   : путь к vocab.json (сохранить из ноутбука: vocab.save(...))
    weights_path : путь к bilstm.pth (torch.save(lstm_model.state_dict(), ...))
    device       : 'cpu' | 'cuda'
    """

    def __init__(self, vocab_path: str, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # 1. Загружаем словарь
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Словарь не найден: {vocab_path}\n"
                "Сохраните из ноутбука командой:\n"
                "    vocab.save('model_artifacts/vocab.json')"
            )
        logger.info("Загружаем словарь: %s", vocab_path)
        self.vocab = Vocabulary.load(vocab_path)
        logger.info("Размер словаря: %d слов", len(self.vocab))

        # 2. Строим архитектуру и загружаем веса
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Веса модели не найдены: {weights_path}\n"
                "Сохраните из ноутбука командой:\n"
                "    torch.save(lstm_model.state_dict(), 'model_artifacts/bilstm.pth')"
            )
        logger.info("Загружаем веса BiLSTM: %s", weights_path)
        self.model = BiLSTMClassifier(vocab_size=len(self.vocab))
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("BiLSTM готов | params=%s | device=%s", f"{n_params:,}", device)

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Определяет эмоцию текста.

        Параметры
        ---------
        text : произвольная строка на английском

        Возвращает
        ----------
        emotion    : название эмоции с наибольшей вероятностью
        confidence : вероятность этой эмоции (0..1)
        all_probs  : словарь {эмоция: вероятность} для всех 6 классов
        """
        # Предобработка — точно как в predict_text_rnn из ноутбука
        cleaned = clean_text(text)
        tokens  = torch.tensor(
            [self.vocab.encode(cleaned, MAX_LEN)],
            dtype=torch.long,
        ).to(self.device)   # (1, 64)

        probs = torch.softmax(self.model(tokens), dim=1).cpu().numpy()[0]  # (6,)

        all_probs  = {EMOTION_MAP[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
        best_idx   = int(probs.argmax())
        emotion    = EMOTION_MAP[best_idx]
        confidence = round(float(probs[best_idx]), 4)

        return emotion, confidence, all_probs
