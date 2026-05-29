import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.data.preprocessing import clean_text
from src.models.bilstm import EMOTION_MAP, NUM_CLASSES, EmotionClassifier
from src.service.schemas import HealthResponse, PredictRequest, PredictResponse

# Загружаем .env 
load_dotenv()

# Логирование 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Глобальное состояние
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель при старте, освобождаем при остановке."""
    vocab_path   = os.getenv("VOCAB_PATH",   "model_artifacts/vocab.json")
    weights_path = os.getenv("MODEL_WEIGHTS", "model_artifacts/bilstm.pth")
    device       = os.getenv("DEVICE",        "cpu")

    logger.info("=" * 55)
    logger.info("  Emotion Analysis Service — старт")
    logger.info("  Словарь : %s", vocab_path)
    logger.info("  Веса    : %s", weights_path)
    logger.info("  Device  : %s", device)
    logger.info("=" * 55)

    try:
        classifier = EmotionClassifier(
            vocab_path=vocab_path,
            weights_path=weights_path,
            device=device,
        )
        _state["classifier"]   = classifier
        _state["vocab_size"]   = len(classifier.vocab)
        _state["device"]       = device
        logger.info("✅ Модель готова к работе")
    except FileNotFoundError as exc:
        logger.error("❌ %s", exc)
        _state["classifier"] = None
        _state["error"]      = str(exc)
        _state["device"]     = device

    yield   # сервер работает

    logger.info("Остановка сервиса...")
    _state.clear()


# ── Приложение ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion Analysis Service",
    description=(
        "Определяет эмоцию текста на английском языке.\n\n"
        "**Модель:** BiLSTM с механизмом внимания "
        "(лучшая модель из ноутбука `emotion_analysis_extended.ipynb`).\n\n"
        "**Поддерживаемые эмоции:** 😢 sadness · 😊 joy · ❤️ love · "
        "😡 anger · 😨 fear · 😲 surprise\n\n"
        "Введите текст через **POST /predict** и получите результат."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    tags=["System"],
)
def health():
    """
    Возвращает **ok** если модель загружена и готова принимать запросы.
    Используйте для мониторинга и Docker HEALTHCHECK.
    """
    if _state.get("classifier") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Модель не загружена. "
                f"Причина: {_state.get('error', 'неизвестно')}. "
                "Проверьте пути VOCAB_PATH и MODEL_WEIGHTS в файле .env."
            ),
        )
    return HealthResponse(
        status="ok",
        model="BiLSTM (embed=128, hidden=256, layers=2, attention)",
        vocab_size=_state["vocab_size"],
        device=_state["device"],
        num_classes=NUM_CLASSES,
        emotions=list(EMOTION_MAP.values()),
    )


# ── GET /emotions ─────────────────────────────────────────────────────────────
@app.get(
    "/emotions",
    summary="Список поддерживаемых эмоций",
    tags=["Info"],
)
def get_emotions():
    """Возвращает все 6 эмоций, которые умеет определять модель."""
    return {
        "emotions": list(EMOTION_MAP.values()),
        "total": NUM_CLASSES,
        "description": {
            "sadness":  "Грусть, печаль, тоска",
            "joy":      "Радость, счастье, восторг",
            "love":     "Любовь, привязанность, нежность",
            "anger":    "Гнев, злость, раздражение",
            "fear":     "Страх, тревога, беспокойство",
            "surprise": "Удивление, изумление",
        },
    }


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Определить эмоцию по тексту",
    tags=["Prediction"],
)
def predict(request: PredictRequest):
    """
    ### Как пользоваться

    1. Нажмите **Try it out**.
    2. В поле `text` введите любую фразу на **английском** языке.
    3. Нажмите **Execute** — получите эмоцию и вероятности.

    ### Примеры фраз

    | Текст | Ожидаемая эмоция |
    |---|---|
    | I am so happy today! | joy |
    | I feel so sad and lonely. | sadness |
    | This makes me so angry! | anger |
    | I'm terrified of what happens next. | fear |
    | I love you with all my heart. | love |
    | Wow, I never expected that! | surprise |

    ### Что происходит внутри

    Текст проходит ту же предобработку, что и при обучении:
    `lower() → удаление URL/тегов → фильтрация символов → encode(max_len=64)`.
    Затем подаётся в BiLSTM → attention → softmax.
    """
    classifier: EmotionClassifier | None = _state.get("classifier")
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не готова. Проверьте GET /health для деталей.",
        )

    text = request.text.strip()
    logger.info("Запрос /predict | text='%.80s...'", text)

    emotion, confidence, all_probs = classifier.predict(text)

    logger.info(
        "Результат | emotion='%s' | confidence=%.4f",
        emotion, confidence,
    )

    return PredictResponse(
        emotion=emotion,
        confidence=confidence,
        all_emotions=all_probs,
        cleaned_text=clean_text(text),
    )
