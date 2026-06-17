# UI Redesign Prompt for rag-lab

You are designing a production-quality web UI for **rag-lab**, a standalone RAG (Retrieval-Augmented Generation) system. The UI must follow Apple Human Interface Guidelines — think SF-style typography, frosted glass, smooth micro-interactions, generous whitespace, rounded corners, and a dark-first aesthetic. No external CDN dependencies. Everything ships in a single HTML template served by a Python FastAPI backend.

---

## 1. What rag-lab does

Users ingest documents (PDF, EPUB, Markdown), the system chunks and embeds them into a vector database, then users ask questions and get cited answers verified by an AI auditor.

Three functional surfaces:

### Ingest
- Upload a file (PDF/EPUB/MD)
- Choose chunking strategy (sentence vs fixed), chunk size, overlap
- See status: parsing → chunking → embedding → done
- Show chunk count on success

### Query
- Type a question
- Tune retrieval parameters (top-K, rerank count, min verifier score, max tokens)
- Optionally enable keyword mode with a regex pattern
- See the answer with inline citations
- See verifier verdict (GROUNDED / PARTIAL / UNGROUNDED / ERROR) with score badge
- Expandable iteration trace showing each refinement round

### Stats
- Live chunk count, database path, embedder model, LLM model
- Auto-refreshes after ingest

---

## 2. Backend API (already built, do not modify)

Base URL: `http://127.0.0.1:8000`

### `POST /api/ingest`
Multipart form upload.

**Request fields:**
- `file` (binary) — the document
- `strategy` (string) — `"sentence"` or `"fixed"`
- `chunk_size` (int) — e.g. `512`
- `overlap` (int) — e.g. `64`

**Success response (200):**
```json
{"status": "ok", "chunks": 14, "filename": "handbook.pdf"}
```

**Error response:**
```json
{"error": "Unsupported file type: .docx"}
```

### `POST /api/query`
JSON body.

**Request body:**
```json
{
  "question": "How many CP is the AI in Business program?",
  "top_k": 20,
  "min_score": 8
}
```

**Success response (200):**
```json
{
  "answer": "The program is 180 CP [abc123-0].",
  "verifier": {
    "score": 10,
    "grounded": true,
    "issues": [],
    "verdict": "GROUNDED"
  },
  "iterations": 1,
  "trace": [
    {
      "iter": 1,
      "query": "How many CP is the AI in Business program?",
      "answer": "The program is 180 CP [abc123-0].",
      "verifier_score": 10,
      "issues": []
    }
  ],
  "partial": false
}
```

**Key response fields:**
- `answer` — may contain `[chunk-id]` citation markers. Render these as subtly styled badges.
- `verifier.score` — integer 1–10. Color-code the badge (green >= 8, yellow 5-7, red < 5).
- `verifier.verdict` — `"GROUNDED"`, `"PARTIAL"`, `"UNGROUNDED"`, `"ERROR"`.
- `partial` — if `true`, max refinement iterations were hit.
- `trace` — array of iteration objects. Render each as a collapsible details row with query, answer, score, issues.
- On error: `{"error": "LLM generation failed: ..."}`

### `GET /api/stats`

**Response:**
```json
{"chunk_count": 3350, "db_path": "./chroma_db"}
```

---

## 3. Apple-like Design Language

Apply these principles:

### Typography
- System font stack: `-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif`
- Use font weights 400 (body), 500 (labels), 600 (headings), 700 (titles)
- `letter-spacing: -0.01em` on headings
- Tabular numbers for stats (`font-variant-numeric: tabular-nums`)

### Color Palette (Dark Mode)
- Background: `#000000` (true black, OLED-friendly)
- Surface cards: `#1c1c1e` (system gray 6)
- Elevated surfaces (modals, dropdowns): `#2c2c2e` (system gray 5) with backdrop blur
- Separators: `#38383a` (system gray 4), 0.5px
- Primary text: `#ffffff` at 92% opacity → `rgba(255,255,255,0.92)`
- Secondary text: `rgba(255,255,255,0.60)`
- Tertiary text (placeholders, disabled): `rgba(255,255,255,0.35)`
- Accent: `#0a84ff` (system blue) — use for primary buttons, links, focus rings
- Green: `#30d158` (success, GROUNDED)
- Orange: `#ff9f0a` (warning, PARTIAL)
- Red: `#ff453a` (error, UNGROUNDED, ERROR)
- File drop zone border: `rgba(255,255,255,0.15)` dashed, `rgba(10,132,255,0.4)` on drag

### Spacing & Layout
- Page max-width: 780px, centered
- Card padding: 20px (horizontal), 16px (vertical)
- Card border-radius: 16px
- Button border-radius: 12px (large), 8px (small)
- Input border-radius: 10px
- Gap between cards: 12px
- Section title: 12px uppercase, `rgba(255,255,255,0.5)`, letter-spacing 0.5px, margin-bottom 10px

### Motion
- Use `cubic-bezier(0.25, 0.1, 0.25, 1.0)` for all transitions (Apple default ease)
- Card hover: subtle scale to 1.002, shadow lift
- Button press: scale to 0.97
- Answer appearance: fade in + slide up 8px over 300ms
- Spinner: 3/4 circle stroke animation, 0.6s duration
- Trace expansion: max-height transition over 250ms

### Components
- **File drop zone**: 120px tall, dashed border, centered SF Symbol-style cloud-arrow-up icon (use inline SVG), "Drop file or click to browse" in secondary text
- **Input fields**: No visible border by default, only a bottom separator line (0.5px `#38383a`). On focus: separator turns accent blue with 0.25s transition
- **Number inputs**: Stepper-style with hidden native spinners, custom +/- buttons on hover
- **Buttons**: 
  - Primary: filled accent blue background, white text, 600 weight
  - Secondary: `rgba(255,255,255,0.1)` background, white text
  - Disabled: 40% opacity
- **Badges**: Pill shape (border-radius 20px), 11px font, 4px 8px padding
  - Score badge: color-coded background at 15% opacity with matching text
  - Chunk citation: monospace font, `rgba(255,255,255,0.08)` background
- **Trace entries**: Collapsible `<details>` with summary showing iteration number + score badge. Content panel with preformatted monospace text.
- **Toast/status bar**: Slides down from top, 3s auto-dismiss, frosted glass background (`backdrop-filter: blur(20px)`)

---

## 4. Implementation Requirements

### Delivery format
A single self-contained HTML file. All CSS in a `<style>` block. All JS in a `<script>` block. No external fonts, icons, or CDN links. SVGs inline. The file replaces the current `PAGE` template string in `rag_lab/web.py`.

### Framework
Vanilla HTML/CSS/JS. No React, Vue, Tailwind, or jQuery. Keep the DOM surface small — the current page has ~100 lines of JS; keep it under 300.

### Responsive
The page must work from 400px to 1200px width. Below 600px, the two-column parameter rows stack vertically. Cards go full-width.

### Accessibility
- All interactive elements are keyboard-focusable (visible focus ring in accent blue)
- File drop zone is tabbable and activates on Enter/Space
- Status messages use `role="status"` for screen reader announcements
- Color is never the sole indicator (verdict text always accompanies the color badge)

### Error states
- Network failure: show a non-dismissible error banner at the top
- Ingest failure: inline error in the ingest card, file drop zone resets
- Query failure: inline error in the answer area
- Empty collection: stats shows "No documents ingested yet" in secondary text
- Long operations: spinner on the active button, button disabled during request

---

## 5. Integration

The backend is a FastAPI app in `rag_lab/web.py`. The current HTML is a Python string constant named `PAGE`. Replace its content with your design. The API endpoints (`/api/ingest`, `/api/query`, `/api/stats`) and their request/response formats are fixed — do not propose changes to them.

The page is served at `GET /`. No routing, no authentication, no sessions. A single user on localhost.

---

## 6. What NOT to do

- Do not add new backend endpoints
- Do not use npm, webpack, or any build tooling
- Do not use Google Fonts, Font Awesome, or any CDN
- Do not add dark/light mode toggle (dark only)
- Do not add chat history, multi-turn conversation, or conversation persistence
- Do not add authentication, user management, or multi-user support
- Do not redesign the API contract
- Do not use CSS frameworks (Tailwind, Bootstrap, etc.)
- Do not exceed 400 lines of JavaScript
