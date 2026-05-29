# Emotion Analysis Service

FastAPI-сервис для определения эмоций по тексту на английском языке.
Модель: **BiLSTM с механизмом внимания** (лучшая модель из `emotion_analysis_extended.ipynb`).
6 эмоций: 😢 sadness · 😊 joy · ❤️ love · 😡 anger · 😨 fear · 😲 surprise

---

## Структура проекта

```
emotion_service/
├── src/
│   ├── data/
│   │   └── preprocessing.py   # clean_text + Vocabulary (из ноутбука)
│   ├── models/
│   │   └── bilstm.py          # архитектура BiLSTM + обёртка инференса
│   └── service/
│       ├── main.py            # FastAPI-приложение, все эндпоинты
│       └── schemas.py         # Pydantic-схемы запросов и ответов
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── report.md
└── README.md
```

---

## Быстрый старт
ПЕРЕЙДИТЕ В ПАПКУ PROJECT
### Шаг 1 — Установите зависимости

```powershell
python -m pip install -r requirements.txt
```

### Шаг 2 — Настройте .env

```powershell
copy .env.example .env
```

### Шаг 3 — Запустите сервис (ТЕКСТ ТОЛЬКО НА АНГЛИЙСКОМ)

```powershell
python -m uvicorn src.service.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте в браузере: **http://localhost:8000/docs**

---

## Эндпоинты

| Метод | Путь        | Описание                                        |
|-------|-------------|-------------------------------------------------|
| GET   | `/health`   | Статус сервиса — модель загружена и готова?     |
| POST  | `/predict`  | Введите текст → получите эмоцию                 |
| GET   | `/emotions` | Список 6 поддерживаемых эмоций с описаниями     |
| GET   | `/docs`     | Swagger UI — попробуйте прямо в браузере        |
| GET   | `/redoc`    | ReDoc документация                              |

---

## Пример запроса через curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am so happy today, everything is going wonderfully!\"}"
```

Ответ:
```json
{
  "emotion": "joy",
  "confidence": 0.9234,
  "all_emotions": {
    "joy": 0.9234,
    "love": 0.0312,
    "surprise": 0.0201,
    "sadness": 0.0121,
    "anger": 0.0088,
    "fear": 0.0044
  },
  "cleaned_text": "i am so happy today everything is going wonderfully"
}
```

---

## Запуск через Docker

```powershell
docker build -t emotion-service .

docker run --rm -p 8000:8000 `
    -v ${PWD}/model_artifacts:/app/model_artifacts `
    --env-file .env `
    emotion-service
```

---

## Демонстрация на защите

1. Запустите сервис (шаги 1–4 выше).
2. Откройте **http://localhost:8000/docs**.
3. `POST /predict` → **Try it out** → введите фразу → **Execute**.
4. Покажите ответ: эмоция, уверенность, вероятности всех 6 классов.
5. Откройте `GET /health` → статус должен быть `"ok"`.
