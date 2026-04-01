# 🔍 AI PDF Assistant

> **Projekt rekrutacyjny** demonstrujący implementację architektury **RAG (Retrieval-Augmented Generation)** z wykorzystaniem **LangChain**, **OpenAI API** oraz **ChromaDB**.

---

## 📌 O projekcie

AI PDF Assistant to aplikacja webowa zbudowana w **Streamlit**, która umożliwia inteligentną rozmowę z dokumentami PDF. Użytkownik wgrywa plik PDF, a system indeksuje jego treść w bazie wektorowej i odpowiada na pytania w języku naturalnym, cytując wyłącznie informacje zawarte w dokumencie.

Projekt powstał jako demonstracja praktycznej znajomości nowoczesnego stosu technologicznego stosowanego w aplikacjach LLM/AI.

---

## 🏗️ Architektura RAG

```
PDF
 │
 ▼
PyPDFLoader                  ← wczytanie dokumentu
 │
 ▼
RecursiveCharacterTextSplitter  ← podział na chunki (1000 zn., overlap 200)
 │
 ▼
OpenAI Embeddings              ← wektoryzacja (text-embedding-3-small)
 │
 ▼
ChromaDB                       ← baza wektorowa (in-memory)
 │
 ▼
RetrievalQA (LangChain)        ← Top-K=5 podobnych chunków
 │
 ▼
GPT-4o Mini                    ← generowanie odpowiedzi
 │
 ▼
Streamlit UI                   ← interfejs użytkownika
```

---

## 🛠️ Stack technologiczny

| Warstwa | Technologia |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain + LangChain Classic |
| LLM | OpenAI GPT-4o Mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB |
| PDF Loader | PyPDFLoader (pypdf) |
| Config | python-dotenv |
| Python | 3.12 |

---

## 🚀 Instalacja i uruchomienie

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/your-username/ai-pdf-assistant.git
cd ai-pdf-assistant
```

### 2. Utwórz środowisko wirtualne z Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 4. Skonfiguruj klucz API

```bash
cp .env.example .env
```

Otwórz `.env` i wpisz swój klucz OpenAI:

```env
OPENAI_API_KEY=sk-...
```

> Klucz API uzyskasz na: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 5. Uruchom aplikację

```bash
streamlit run app.py
```

Aplikacja otworzy się automatycznie pod adresem `http://localhost:8501`.

---

## 📂 Struktura projektu

```
ai-pdf-assistant/
├── app.py              ← główna aplikacja Streamlit
├── requirements.txt    ← zależności Python
├── .env.example        ← szablon zmiennych środowiskowych
├── .env                ← klucz API (nie commitować!)
└── README.md
```

---

## ⚙️ Jak działa aplikacja

1. **Upload PDF** — użytkownik wgrywa plik przez sidebar
2. **Chunking** — dokument jest dzielony na fragmenty po 1000 znaków z 200-znakowym overlappem
3. **Embedding** — każdy chunk jest zamieniany na wektor liczbowy przez model `text-embedding-3-small`
4. **Indeksowanie** — wektory są przechowywane w ChromaDB (in-memory)
5. **Pytanie** — zapytanie użytkownika jest również wektoryzowane
6. **Retrieval** — system pobiera 5 najbardziej semantycznie podobnych chunków
7. **Generowanie** — GPT-4o Mini generuje odpowiedź na podstawie pobranych fragmentów
8. **Historia** — rozmowa jest przechowywana w `st.session_state`

---

## 🔑 Zmienne środowiskowe

| Zmienna | Opis | Wymagana |
|---|---|---|
| `OPENAI_API_KEY` | Klucz API OpenAI | ✅ |

---

## 📋 Wymagania systemowe

- Python **3.12** (3.11+ zalecane; 3.14 nieobsługiwany przez zależności)
- pip 23+
- Konto OpenAI z dostępem do API

---

## 💡 Możliwe rozszerzenia

- [ ] Obsługa wielu PDF jednocześnie
- [ ] Trwały vector store (zapis ChromaDB na dysk)
- [ ] Eksport historii czatu do PDF/TXT
- [ ] Wybór modelu LLM przez użytkownika
- [ ] Wskazywanie strony źródłowej w odpowiedzi
- [ ] Obsługa plików DOCX i TXT

---

## 👩‍💻 Autor

Stworzone jako projekt rekrutacyjny demonstrując znajomość:
- Architektury RAG i jej praktycznej implementacji
- LangChain (chains, retrievers, prompts)
- OpenAI API (chat + embeddings)
- Streamlit (session state, custom UI, dark theme)
- ChromaDB jako bazy wektorowej

---

## 📄 Licencja

MIT
