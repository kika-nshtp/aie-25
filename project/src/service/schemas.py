from typing import Dict
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Тело запроса POST /predict — просто текст."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст на АНГЛИЙСКОМ языке, эмоцию которого нужно определить",
        examples=["I am so happy today, everything is going wonderfully!"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I am so happy today, everything is going wonderfully!"
            }
        }
    }


class PredictResponse(BaseModel):
    """Ответ POST /predict."""
    emotion: str = Field(
        ...,
        description="Определённая эмоция: sadness / joy / love / anger / fear / surprise",
        examples=["joy"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность модели (0 = нет уверенности, 1 = абсолютная)",
        examples=[0.9234],
    )
    all_emotions: Dict[str, float] = Field(
        ...,
        description="Вероятности всех 6 эмоций",
        examples=[{"joy": 0.9234, "love": 0.0312, "surprise": 0.0201,
                   "sadness": 0.0121, "anger": 0.0088, "fear": 0.0044}],
    )
    cleaned_text: str = Field(
        ...,
        description="Текст после предобработки (как видит его модель)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "emotion": "joy",
                "confidence": 0.9234,
                "all_emotions": {
                    "joy": 0.9234, "love": 0.0312, "surprise": 0.0201,
                    "sadness": 0.0121, "anger": 0.0088, "fear": 0.0044,
                },
                "cleaned_text": "i am so happy today everything is going wonderfully",
            }
        }
    }


class HealthResponse(BaseModel):
    """Ответ GET /health."""
    status: str       = Field(..., description="'ok' если сервис готов к работе")
    model: str        = Field(..., description="Архитектура модели")
    vocab_size: int   = Field(..., description="Размер словаря")
    device: str       = Field(..., description="cpu / cuda")
    num_classes: int  = Field(..., description="Количество эмоций (6)")
    emotions: list    = Field(..., description="Список всех поддерживаемых эмоций")
