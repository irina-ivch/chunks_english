import os
import re
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Response, Request, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session as DBSession
from sqlalchemy.sql import func
from passlib.context import CryptContext
from jose import jwt, JWTError

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-5"
SECRET_KEY = os.getenv("SECRET_KEY", "please-change-this-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DictionaryItem(Base):
    __tablename__ = "dictionary_items"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    chunk = Column(String, nullable=False)
    card_json = Column(Text, nullable=False)
    video_id = Column(String, nullable=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now())


# ── Auth helpers ──────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: DBSession = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
    return db.query(User).filter(User.id == user_id).first()


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Необходима авторизация")
    return user


# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set")
    yield


app = FastAPI(title="chunks_english API", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY не настроен")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise HTTPException(status_code=400, detail="Не удалось извлечь ID видео из ссылки")


def strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw


def build_system_prompt(
    level: str,
    video_topic: str,
    chunk_list: str,
    grammar_titles: str,
    lang: str,
) -> str:
    if lang == "en":
        opening = (
            f'Hey! 👋 I\'ve already gone through the video about "{video_topic}" '
            f'and put together a lesson for you. There are some great chunks and grammar points '
            f'— I think you\'ll enjoy it. Shall we follow my plan or is there something '
            f'specific you\'d like to work on first?'
        )
        lang_instruction = (
            "Conduct the entire lesson in English. Give error explanations in English too, "
            "but simplify if the student seems confused."
        )
    else:
        opening = (
            f'Привет! 👋 Я уже изучил видео про "{video_topic}" и подготовил для тебя урок. '
            f'Есть классные чанки и грамматика — думаю, понравится. '
            f'Начнём по моему плану или сначала расскажешь что хочешь поработать?'
        )
        lang_instruction = (
            "Задания и переписку веди на английском. "
            "Объяснения ошибок, оценки и подсказки давай на русском. "
            "Если ученик просит объяснить — объясняй на русском."
        )

    return f"""Ты — дружелюбный и проактивный репетитор английского языка по имени Лингва.

КОНТЕКСТ СЕССИИ:
- Уровень ученика: {level}
- Тема видео: {video_topic}
- Изученные чанки: {chunk_list}
- Грамматические конструкции из видео: {grammar_titles}

ПЕРВОЕ СООБЩЕНИЕ: если ты получаешь системный триггер __START__, ответь ТОЛЬКО этим приветствием, без изменений:
{opening}

ЕСЛИ УЧЕНИК ГОВОРИТ "начнём" / "по плану" / "давай" / "let's start" / "go ahead":
Веди структурированный урок, чередуя задания ниже. Сам выбираешь порядок. После каждого ответа:
- если ответ правильный: коротко похвали (1 предложение) и в том же сообщении сразу дай следующее задание — не жди реакции
- если ответ неправильный или неполный: исправь → объясни ПОЧЕМУ именно так (логика языка, а не просто правило) → дай формулу конструкции в формате [hl]subject + had + past participle[/hl] → покажи исправленное предложение где нужная часть обёрнута в [hl]...[/hl] → дай подсказку как запомнить → дай ещё одну попытку
- никогда не заканчивай сообщение без нового задания или конкретного вопроса ученику
- никогда не говори "последнее задание", "финальное задание" и подобное — заданий всегда достаточно
- каждые 5 выполненных упражнений предложи: "Кстати, советую пересмотреть видео — теперь ты будешь слышать эти чанки по-другому. Потом возвращайся, продолжим!" и жди ответа перед следующим заданием

ЕСЛИ УЧЕНИК ХОЧЕТ ОБСУДИТЬ ЧТО-ТО СВОЁ:
Поддержи тему, активно используй изученные чанки в своих репликах и мягко исправляй ошибки по ходу.

ТИПЫ ЗАДАНИЙ (чередуй, не объявляй тип вслух):
1. GAP-FILL — дай 2-5 предложений с пропусками, куда нужно вставить правильный чанк
2. SITUATION RECALL — опиши ситуацию, ученик должен сам вспомнить подходящий чанк
3. GRAMMAR DRILL — отрабатывай грамматические конструкции из видео, вплетай чанки, давай подсказки если много ошибок
4. TRANSLATION — предложи перевести фразу на английский, чанки вплетай органично
5. ERROR CORRECTION — дай предложение с неправильно использованным чанком, ученик находит и исправляет
6. REFORMULATION — дай предложение без чанка, ученик перефразирует используя подходящий из изученных (без подсказки какой)
7. MINI-CHAT — начни переписку до 10 сообщений. Определи тональность по теме видео (рабочая/повседневная/другая) и играй роль подходящего собеседника. Ученик органично использует изученные чанки.

ФОРМАТ ОТВЕТОВ:
- Никогда не используй markdown: никаких **, __, ##, --- и прочих символов разметки
- Пиши простым текстом
- Каждый ответ — один смысловой блок: либо обратная связь по заданию, либо новое задание. Никогда не совмещай оба в одном сообщении
- Длина ответа — не более 100 слов
- Без числовых оценок (не пиши "8/10" или подобное)

ЯЗЫК: {lang_instruction}"""


# ── Request / Response models ─────────────────────────────────────────────────

class TranscriptRequest(BaseModel):
    url: str

class TranscriptResponse(BaseModel):
    video_id: str
    transcript: str
    language: str

class AnalyzeRequest(BaseModel):
    transcript: str
    level: str

class ChunkItem(BaseModel):
    chunk: str
    type: str
    level: str
    original_sentence: str
    meaning_ru: str
    register: str
    why_useful: str
    similar_chunks: list[str]

class AnalyzeResponse(BaseModel):
    chunks: list[ChunkItem]

class GenerateTheoryRequest(BaseModel):
    transcript: str
    level: str
    chunks: list[ChunkItem]
    video_id: str
    lesson_language: str = "ru"

class ChunkTheory(BaseModel):
    chunk: str
    type: str
    level: str
    meaning_ru: str
    register: str
    why_useful: str
    similar_chunks: list[str]
    original_sentence: str
    other_examples: list[str]

class GrammarItem(BaseModel):
    title: str
    explanation_ru: str
    structure: str
    examples: list[str]

class GenerateTheoryResponse(BaseModel):
    chunks: list[ChunkTheory]
    grammar: list[GrammarItem]
    system_prompt: str
    video_topic: str

class ChatRequest(BaseModel):
    messages: list[dict]
    system_prompt: str

class ChatResponse(BaseModel):
    reply: str

# ── Auth models ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    session_words: list[dict] = []

class LoginRequest(BaseModel):
    email: str
    password: str

class SyncWordsRequest(BaseModel):
    words: list[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def register(req: RegisterRequest, response: Response, db: DBSession = Depends(get_db)):
    if not req.email or "@" not in req.email:
        raise HTTPException(status_code=400, detail="Некорректный email")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Пароль должен быть не менее 6 символов")
    if db.query(User).filter(User.email == req.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

    user = User(email=req.email.lower(), password_hash=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    # Migrate session words
    for word in req.session_words:
        chunk = word.get("chunk")
        if chunk and not db.query(DictionaryItem).filter(
            DictionaryItem.user_id == user.id,
            DictionaryItem.chunk == chunk,
        ).first():
            db.add(DictionaryItem(
                user_id=user.id,
                chunk=chunk,
                card_json=json.dumps(word, ensure_ascii=False),
                video_id=word.get("videoId"),
            ))
    db.commit()

    token = create_token(user.id)
    response.set_cookie(
        "access_token", token,
        httponly=True, max_age=TOKEN_EXPIRE_DAYS * 86400, samesite="lax",
    )
    return {"id": user.id, "email": user.email}


@app.post("/api/auth/login")
async def login(req: LoginRequest, response: Response, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email.lower()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    token = create_token(user.id)
    response.set_cookie(
        "access_token", token,
        httponly=True, max_age=TOKEN_EXPIRE_DAYS * 86400, samesite="lax",
    )
    return {"id": user.id, "email": user.email}


@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user: Optional[User] = Depends(get_current_user)):
    if not user:
        return {"user": None}
    return {"user": {"id": user.id, "email": user.email}}


# ── Dictionary ────────────────────────────────────────────────────────────────

@app.get("/api/dictionary")
async def get_dictionary(user: User = Depends(require_user), db: DBSession = Depends(get_db)):
    items = (
        db.query(DictionaryItem)
        .filter(DictionaryItem.user_id == user.id)
        .order_by(DictionaryItem.added_at.desc())
        .all()
    )
    result = []
    for item in items:
        try:
            card = json.loads(item.card_json)
        except Exception:
            card = {"chunk": item.chunk}
        card["id"] = item.id
        card["addedAt"] = int(item.added_at.timestamp() * 1000) if item.added_at else 0
        result.append(card)
    return result


@app.post("/api/dictionary/sync")
async def sync_dictionary(
    req: SyncWordsRequest,
    user: User = Depends(require_user),
    db: DBSession = Depends(get_db),
):
    added = 0
    for word in req.words:
        chunk = word.get("chunk")
        if chunk and not db.query(DictionaryItem).filter(
            DictionaryItem.user_id == user.id,
            DictionaryItem.chunk == chunk,
        ).first():
            db.add(DictionaryItem(
                user_id=user.id,
                chunk=chunk,
                card_json=json.dumps(word, ensure_ascii=False),
                video_id=word.get("videoId"),
            ))
            added += 1
    db.commit()
    return {"added": added}


@app.delete("/api/dictionary/{item_id}")
async def delete_dictionary_item(
    item_id: int,
    user: User = Depends(require_user),
    db: DBSession = Depends(get_db),
):
    item = db.query(DictionaryItem).filter(
        DictionaryItem.id == item_id,
        DictionaryItem.user_id == user.id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Не найдено")
    db.delete(item)
    db.commit()
    return {"ok": True}


# ── Transcript ────────────────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> tuple[str, str]:
    """Fetch transcript via Supadata API."""
    api_key = os.getenv("SUPADATA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SUPADATA_API_KEY не настроен")

    resp = httpx.get(
        "https://api.supadata.ai/v1/youtube/transcript",
        headers={"x-api-key": api_key},
        params={"videoId": video_id, "text": "true"},
        timeout=30,
    )

    if resp.status_code == 404:
        raise HTTPException(status_code=422, detail="Субтитры не найдены для этого видео.")
    if resp.status_code != 200:
        detail = resp.json().get("message", resp.text)
        raise HTTPException(status_code=422, detail=f"Не удалось загрузить субтитры: {detail}")

    data = resp.json()
    text = data.get("content", "")
    lang = data.get("lang", "en")
    return text, lang


@app.post("/api/transcript", response_model=TranscriptResponse)
async def get_transcript(req: TranscriptRequest):
    video_id = extract_video_id(req.url)
    text, lang_code = fetch_transcript(video_id)

    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000]) + " ..."

    return TranscriptResponse(video_id=video_id, transcript=text, language=lang_code)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_chunks(req: AnalyzeRequest):
    client = get_client()

    word_count = len(req.transcript.split())
    if word_count < 1500:
        qty = "5–10"
    elif word_count < 5000:
        qty = "10–20"
    else:
        qty = "15–20"

    prompt = f"""You are a lexical chunk extraction expert for an English learning app called chunks_english.

The learner's CEFR level is: {req.level}

You will receive a YouTube video transcript. Extract the most valuable lexical chunks that are ABOVE the learner's level but reachable with effort.

## What counts as a chunk
Extract ONLY multi-word units that native speakers use as ready-made blocks:
- Fixed expressions: "as a matter of fact", "on top of that"
- Semi-fixed frames: "I was wondering", "what tends to happen is"
- Collocations: "make a decision", "raise awareness", "deeply concerned"
- Phrasal verbs in context: "figure out", "end up", "come across"
- Discourse markers: "having said that", "that being said"
- Idiomatic phrases: "game changer", "on the same page"

## What NOT to extract
- Single words (even rare ones)
- Technical jargon specific only to this video topic
- Proper nouns, brand names
- Filler words: "you know", "I mean", "like"
- Chunks already at or below level {req.level}

## Chunk length — IMPORTANT
- Extract 2–5 words maximum per chunk
- For semi_fixed frames: extract only the fixed core, not the full clause
  ✓ "let me know" — not "let me know what I've missed"
  ✓ "I was wondering" — not "I was wondering if you could help"
- For collocations: extract the core pair or triple only
  ✓ "send it out", "double-check" — not "double-check before sending it out"
- If a chunk appears with a tail in the transcript, strip the tail

## Selection criteria (apply all three)
1. TRANSFERABILITY — Can this chunk appear in many different conversations?
2. NATURALNESS — Would replacing one word make it sound unnatural?
3. LEVEL FIT — Is it above {req.level} but learnable in one session?

## Output
Return a JSON array only, no markdown fences, no extra text:
[
  {{
    "chunk": "on top of that",
    "type": "discourse_marker",
    "level": "B2",
    "original_sentence": "<exact sentence from the transcript containing this chunk>",
    "meaning_ru": "<Russian translation/explanation>",
    "register": "neutral",
    "why_useful": "<one sentence in Russian: why an intermediate learner should know this>",
    "similar_chunks": ["<similar chunk 1>", "<similar chunk 2>"]
  }}
]

Types: fixed_expression | collocation | phrasal_verb | discourse_marker | idiom | semi_fixed
Register: formal | neutral | informal

Extract {qty} chunks. Prioritise transferability over quantity.

## Transcript:
{req.transcript[:4000]}
"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = strip_json_fences(message.content[0].text)

    try:
        data = json.loads(raw)
        chunks = [ChunkItem(**item) for item in data]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать ответ модели: {e}\n\nRaw: {raw[:300]}",
        )

    return AnalyzeResponse(chunks=chunks)


@app.post("/api/generate-theory", response_model=GenerateTheoryResponse)
async def generate_theory(req: GenerateTheoryRequest):
    client = get_client()

    chunks_json = json.dumps(
        [c.model_dump() for c in req.chunks], ensure_ascii=False, indent=2
    )
    chunk_list = ", ".join(f'"{c.chunk}" ({c.meaning_ru})' for c in req.chunks)

    prompt = f"""Ты — методист по английскому языку. Уровень ученика: {req.level}.

Ученик выбрал для изучения следующие лексические чанки из YouTube видео. Твоя задача — обогатить каждый чанк тремя живыми примерами и извлечь грамматические конструкции.

ЧАНКИ ДЛЯ ИЗУЧЕНИЯ:
{chunks_json}

ТРАНСКРИПТ ВИДЕО (для контекста):
{req.transcript[:3000]}

Верни ТОЛЬКО валидный JSON без markdown-оберток:
{{
  "video_topic": "<тема видео в 5-7 словах на русском, например: прокрастинация и как с ней бороться>",
  "chunks": [
    {{
      "chunk": "<чанк точно как в списке>",
      "type": "<тип из исходных данных>",
      "level": "<уровень из исходных данных>",
      "meaning_ru": "<перевод/объяснение из исходных данных>",
      "register": "<регистр из исходных данных>",
      "why_useful": "<why_useful из исходных данных>",
      "similar_chunks": ["<similar из исходных данных>"],
      "original_sentence": "<предложение из транскрипта — из исходных данных>",
      "other_examples": [
        "<живой пример использования в реальной речи 1>",
        "<живой пример использования в реальной речи 2>",
        "<живой пример использования в реальной речи 3>"
      ]
    }}
  ],
  "grammar": [
    {{
      "title": "<название грамматической конструкции>",
      "explanation_ru": "<объяснение на русском, 2-3 предложения, просто и понятно>",
      "structure": "<формула, например: subject + have/has + past participle>",
      "examples": [
        "<пример из транскрипта видео>",
        "<ещё пример из транскрипта>",
        "<самостоятельный пример>"
      ]
    }}
  ]
}}

Верни ровно {len(req.chunks)} карточек чанков и ровно 3 грамматические конструкции которые реально встречаются в транскрипте.
"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = strip_json_fences(message.content[0].text)

    try:
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка разбора теории: {e}\n\nRaw: {raw[:400]}",
        )

    video_topic = data.get("video_topic", "")
    grammar_titles = ", ".join(g.get("title", "") for g in data.get("grammar", []))

    system_prompt = build_system_prompt(
        level=req.level,
        video_topic=video_topic,
        chunk_list=chunk_list,
        grammar_titles=grammar_titles,
        lang=req.lesson_language,
    )

    return GenerateTheoryResponse(
        chunks=[ChunkTheory(**c) for c in data.get("chunks", [])],
        grammar=[GrammarItem(**g) for g in data.get("grammar", [])],
        system_prompt=system_prompt,
        video_topic=video_topic,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    client = get_client()

    messages = req.messages if req.messages else [{"role": "user", "content": "__START__"}]

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=req.system_prompt,
        messages=messages,
    )

    return ChatResponse(reply=message.content[0].text)
