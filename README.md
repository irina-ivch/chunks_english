# LinguaFrame 🎬

Персональный репетитор английского через YouTube — изучай язык на реальных видео.

## Возможности

- Извлечение субтитров из любого YouTube видео
- Анализ словарного запаса по уровню CEFR (A1–C2)
- Интерактивный урок по методике Chunking + Sentence Mining
- Живой чат с AI-репетитором
- Личный словарь (сохраняется между сессиями)
- Режим повторения для уже изученных видео

## Требования

- Python 3.11+
- Ключ Anthropic API

## Установка и запуск

### 1. Клонируй или распакуй проект

```bash
cd linguaframe
```

### 2. Создай виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# или
venv\Scripts\activate         # Windows
```

### 3. Установи зависимости

```bash
pip install -r requirements.txt
```

### 4. Настрой API ключ

Скопируй `.env.example` в `.env` и вставь свой ключ:

```bash
cp .env.example .env
```

Отредактируй `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Либо передай ключ через переменную окружения:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Запусти сервер

```bash
uvicorn main:app --reload --port 8000
```

Открой браузер: **http://localhost:8000**

## Структура проекта

```
linguaframe/
├── main.py          # FastAPI бэкенд
├── index.html       # Весь фронтенд (один файл)
├── requirements.txt
├── .env.example
└── README.md
```

## API endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Фронтенд |
| POST | `/api/transcript` | Извлечь субтитры YouTube |
| POST | `/api/analyze` | Проанализировать слова по уровню |
| POST | `/api/generate-lesson` | Сгенерировать урок |
| POST | `/api/chat` | Сообщение репетитору |

## Советы по выбору видео

- **Ted Talks** — структурированная речь, хорошие субтитры
- **BBC Learning English** — специально для изучающих
- **Подкасты с субтитрами** — natural speech
- **Влоги** — разговорный английский

> Видео должно иметь английские субтитры (авто-сгенерированные тоже подходят).
