from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    url: str = ""
    text: str = ""
    model: str = ""
    source_type: str = ""  # "video", "article", "pdf" — auto-detected if empty


class Flashcard(BaseModel):
    q: str
    a: str


class Timestamp(BaseModel):
    time: str
    label: str


class VocabTerm(BaseModel):
    term: str
    definition: str


class AnkiExportRequest(BaseModel):
    title: str = "PodMemory Export"
    flashcards: list[Flashcard]


class AnalysisResponse(BaseModel):
    title: str
    tldr: str
    key_insights: list[str]
    flashcards: list[Flashcard]
    timestamps: list[Timestamp] = []
    action_items: list[str] = []
    tags: list[str] = []
    difficulty: str = "intermediate"
    source_type: str = ""  # "video", "article", "pdf", "text"
    vocabulary: list[VocabTerm] = []  # key terms + definitions (for articles/books)
    quotes: list[str] = []  # notable quotes worth remembering
    model_used: str = ""
    transcript_source: str = ""
    transcript_language: str = ""
    transcript_length: int = 0
