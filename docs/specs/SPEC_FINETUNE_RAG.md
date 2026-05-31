# SPEC_FINETUNE_RAG — Fine-tune Pipeline + Universal RAG (UPDATE 3)

Implement two major features for the qa-agent system:
PART A: Fine-tuning Pipeline with Synthetic Data
PART B: Universal RAG Pipeline (Playwright Docs + URL Crawling + Document Parsing)

---

## PART A: FINE-TUNING PIPELINE

### A1: src/data/synthetic_gen.py — Synthetic Dataset Generator

Update generate_synthetic_dataset() to produce high-quality training data from all 13 domains.

The generation pipeline must work in 3 phases:

PHASE 1 — Generate examples per domain
For each domain in DOMAIN_REGISTRY:
  For each scenario in domain["scenarios"]:
    Call LLM with this prompt structure:

    System: "You are an expert QA engineer. Generate a realistic Playwright Python test
    for the given scenario. Output ONLY raw Python code using pytest-playwright format.
    Use only: get_by_role(), get_by_label(), get_by_text(), get_by_test_id(), expect().
    Function name must start with test_. No markdown fences. No imports except pytest
    and playwright.sync_api."

    User: "Domain: {domain_name}
    Scenario: {scenario}
    Write a complete pytest-playwright test function for this scenario."

    Save as ChatML JSONL entry:
    {
      "messages": [
        {"role": "system", "content": "<system prompt above>"},
        {"role": "user", "content": "Domain: {domain}\nScenario: {scenario}"},
        {"role": "assistant", "content": "<generated pytest code>"}
      ],
      "metadata": {
        "domain": "{domain_name}",
        "scenario": "{scenario}",
        "type": "happy_path"
      }
    }

  For each item in domain["edge_cases"]:
    Same process but type = "edge_case"
    Add to user prompt: "This is an edge case test. Include appropriate error handling
    and verify the correct error message or behavior."

PHASE 2 — Healing examples (critical for self-healing training)
Generate 50 healing pair examples across all domains:
  Each example shows: broken locator → correct locator
  Format:
  {
    "messages": [
      {"role": "system", "content": "You are a Playwright self-healing expert.
       Given a broken locator and current AxTree, return the correct locator.
       Output ONLY the corrected single line of Python code."},
      {"role": "user", "content": "Broken: {broken_line}\nAxTree:\n{sample_axtree}"},
      {"role": "assistant", "content": "{fixed_line}"}
    ],
    "metadata": {"domain": "{domain}", "type": "healing"}
  }

  Use these broken→fixed pairs as templates (generate variations):
  - page.click('#submit-btn') → page.get_by_role('button', name='Submit').click()
  - page.fill('#email', value) → page.get_by_label('Email').fill(value)
  - page.locator('.error-msg').text → expect(page.get_by_role('alert')).to_be_visible()
  - page.find_element('xpath=//button') → page.get_by_role('button', name='Login').click()

PHASE 3 — Validation and quality filter
After generating all examples:
  - Parse each "assistant" content as Python using ast.parse()
  - If SyntaxError: discard the example and log warning
  - If no test_ function found: discard
  - If CSS selector or XPath found (regex: r'[#.]\w+|xpath='): discard
  - Log final stats: total generated, total discarded, total saved
  - Save valid examples to ~/.qa-agent/datasets/synthetic_universal.jsonl

Add progress bar using rich.progress during generation.
Add resume capability: if output file exists, skip already-generated scenarios.

---

### A2: src/finetune/trainer.py — QLoRA Training Pipeline

Complete implementation with 6GB VRAM safety:

class QLoRATrainer:

  def check_vram(self) -> float:
    Run: nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    Parse output and return free VRAM in GB
    If free VRAM < 5.5: raise RuntimeError with clear message:
    "Insufficient VRAM: {X}GB free, need 5.5GB minimum.
     Close other GPU applications and retry."

  def load_model(self):
    Use unsloth FastLanguageModel.from_pretrained():
      model_name = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
      max_seq_length = 2048
      dtype = None (auto-detect)
      load_in_4bit = True

    Apply LoRA with get_peft_model():
      r = 16
      lora_alpha = 32
      target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
      lora_dropout = 0.05
      bias = "none"
      use_gradient_checkpointing = "unsloth"

  def prepare_dataset(self, jsonl_path: str):
    Load JSONL file
    Convert ChatML messages to Unsloth format using:
    tokenizer.apply_chat_template(messages, tokenize=False)
    Filter examples longer than max_seq_length tokens
    Log: total examples, filtered out, final count
    Return HuggingFace Dataset object

  def train(self, dataset):
    Use SFTTrainer with these args:
      per_device_train_batch_size = 1
      gradient_accumulation_steps = 4
      warmup_steps = 10
      max_steps = 300  (enough for 6GB VRAM constraint)
      learning_rate = 2e-4
      fp16 = True  (use fp16 not bf16 for broader GPU support)
      logging_steps = 10
      output_dir = ~/.qa-agent/checkpoints/
      save_steps = 50

    Log training progress every 10 steps:
    "Step {step}/300 | Loss: {loss:.4f} | LR: {lr:.2e}"

    Save final checkpoint when done

  def run(self, jsonl_path: str):
    self.check_vram()
    self.load_model()
    dataset = self.prepare_dataset(jsonl_path)
    self.train(dataset)
    logger.info("Training complete. Run finetune --export to create Ollama model.")

---

### A3: src/finetune/export.py — GGUF Export and Ollama Registration

class ModelExporter:

  def merge_and_export(self, checkpoint_dir: str):

    Step 1 — Merge LoRA weights:
    Use unsloth FastLanguageModel.from_pretrained(checkpoint_dir)
    Save merged model to ~/.qa-agent/merged_model/

    Step 2 — Export to GGUF Q4_K_M:
    model.save_pretrained_gguf(
      "~/.qa-agent/gguf/",
      tokenizer,
      quantization_method="q4_k_m"
    )
    Output file: ~/.qa-agent/gguf/qa-agent-coder-q4_k_m.gguf

    Step 3 — Create Ollama Modelfile:
    Write to ~/.qa-agent/gguf/Modelfile:
    """
    FROM ./qa-agent-coder-q4_k_m.gguf
    SYSTEM "You are an expert Playwright QA engineer. You write pytest-playwright tests
    using only semantic locators: get_by_role, get_by_label, get_by_text, get_by_test_id.
    You never use CSS selectors or XPath. You always use expect() for assertions."
    PARAMETER temperature 0.1
    PARAMETER top_p 0.9
    PARAMETER num_ctx 4096
    """

    Step 4 — Register in Ollama:
    Run: ollama create qa-agent-coder -f ~/.qa-agent/gguf/Modelfile

    Step 5 — Smoke test:
    Send test prompt to qa-agent-coder via Ollama API
    Verify response contains "def test_"
    Log: "Model qa-agent-coder registered and verified OK"

---

### A4: main.py — Finetune Mode Handler

Update --mode finetune handler to support sub-steps via --step flag:

--step generate  → run synthetic dataset generation only
--step train     → run QLoRA training on existing dataset
--step export    → merge + export to GGUF + register in Ollama
--step all       → run all three steps sequentially (default)

Commands:
uv run python main.py --mode finetune --step generate
uv run python main.py --mode finetune --step train --dataset ~/.qa-agent/datasets/synthetic_universal.jsonl
uv run python main.py --mode finetune --step export --checkpoint ~/.qa-agent/checkpoints/
uv run python main.py --mode finetune --step all

After export completes successfully, update config/agent.yaml automatically:
Change llm.model from "qwen2.5-coder:7b-instruct-q4_K_M" to "qa-agent-coder"

---

## PART B: UNIVERSAL RAG PIPELINE

### B1: src/rag/ingestion.py — Multi-Source Ingestion Pipeline

Implement 3 ingestion sources with a unified interface:

class RAGIngestionPipeline:

  SOURCE 1 — Playwright Documentation Ingester

  Method: async def ingest_playwright_docs(self)

  Fetch these URLs using httpx and parse with BeautifulSoup:
  urls = [
    "https://playwright.dev/python/docs/api/class-page",
    "https://playwright.dev/python/docs/api/class-locator",
    "https://playwright.dev/python/docs/api/class-expect",
    "https://playwright.dev/python/docs/locators",
    "https://playwright.dev/python/docs/best-practices",
    "https://playwright.dev/python/docs/assertions",
    "https://playwright.dev/python/docs/auth",
    "https://playwright.dev/python/docs/pages",
  ]

  For each URL:
    Fetch HTML with httpx (timeout=30s, retry 3 times)
    Extract main content: soup.select_one('article') or soup.select_one('main')
    Remove nav, footer, script, style tags
    Convert to clean markdown using markdownify library
    Chunk by h2/h3 headers (each section = one chunk)
    Each chunk metadata: {source: "playwright_docs", url: url, section: heading_text}
    Upsert to LanceDB with embedding from nomic-embed-text

  Log: "Playwright docs ingested: {N} chunks from {M} pages"

  SOURCE 2 — URL Crawler (for target web app)

  Method: async def ingest_url(self, url: str, depth: int = 2)

  Use Playwright headless browser to crawl the target app:

  Step 1 — Discovery:
    Visit the URL
    Extract all internal links (same domain only): page.eval_on_selector_all('a[href]', ...)
    Filter out: images, PDFs, external domains, #anchors
    Deduplicate URLs
    Limit to max 50 pages total

  Step 2 — For each discovered page:
    Navigate to page
    Extract AxTree using page.aria_snapshot()
    Extract page title and meta description
    Extract all visible text content
    Extract all form labels, button names, input placeholders
    Build structured context:
    {
      "url": page_url,
      "title": page_title,
      "axtree_summary": pruned_axtree,
      "interactive_elements": [list of roles+names],
      "forms": [list of form field labels],
    }
    Convert to markdown chunk
    Metadata: {source: "url_crawl", url: page_url, domain: detected_domain}
    Upsert to LanceDB

  Step 3 — Rate limiting:
    Wait 1 second between pages
    Skip pages that return 404 or 403
    Log progress: "Crawled {N}/{total} pages"

  SOURCE 3 — Document Parser (PDF, Word, Markdown, Text)

  Method: async def ingest_document(self, file_path: str)

  Detect file type from extension:

  .pdf:
    Use pypdf library: PdfReader(file_path)
    Extract text page by page
    If text is empty (scanned PDF): log warning "Scanned PDF detected, text extraction limited"

  .docx:
    Use python-docx library: Document(file_path)
    Extract paragraphs and tables
    Convert tables to markdown table format

  .md or .txt:
    Read directly as text

  .json or .yaml:
    Parse as structured data
    Convert to readable markdown description

  After extraction for all types:
    Clean text: remove excessive whitespace, fix encoding issues
    Semantic chunking: split by headers (## , ###) or by paragraph blocks of ~500 tokens
    Each chunk must have minimum 50 characters (skip tiny chunks)
    Metadata: {source: "document", file: filename, type: extension, page: page_number}
    Upsert to LanceDB with embedding

  Log: "Document ingested: {filename} → {N} chunks"

  UNIFIED METHOD:

  async def ingest_all(self, sources: dict):
    sources format:
    {
      "playwright_docs": True,
      "urls": ["https://app1.com", "https://app2.com"],
      "documents": ["path/to/spec.pdf", "path/to/requirements.docx"]
    }

    Run all ingestion in sequence with progress tracking
    Print summary table at end:
    | Source | Type | Chunks Added |
    |---|---|---|
    | playwright_docs | web | 245 |
    | https://app.com | crawl | 38 |
    | requirements.pdf | document | 22 |
    | Total | | 305 |

---

### B2: src/rag/retriever.py — Enhanced Universal Retriever

Update search() method to support source filtering:

async def search(
  self,
  query: str,
  top_k: int = 5,
  sources: list[str] = None,  # filter by source type
  domain: str = None           # filter by domain metadata
) -> list[RAGChunk]:

  Build filter expression if sources or domain provided:
  filter_expr = None
  if sources:
    filter_expr = f"source IN {sources}"
  if domain:
    filter_expr = f"metadata_json LIKE '%{domain}%'"

  Run hybrid search:
  1. Semantic search with filter_expr
  2. BM25 keyword search on content field
  3. Reciprocal Rank Fusion with weights: semantic=0.7, bm25=0.3
  4. Return top_k results

Add method: async def get_playwright_context(self, locator_type: str) -> str
  Query specifically for playwright docs about the locator_type
  Used by generator to get accurate API usage before writing code
  Example: get_playwright_context("get_by_role") returns relevant docs

Add method: async def get_page_context(self, url: str) -> str
  Query for crawled pages matching the URL
  Returns AxTree and interactive elements for that specific URL
  Used by planner to understand page structure before writing test plan

---

### B3: src/agents/planner.py — RAG-Enhanced Planning

Update PlannerAgent.plan() to use RAG context:

Before generating test plan:
1. Call retriever.get_page_context(url) → get page structure from RAG
2. Call retriever.search(requirement, sources=["playwright_docs"]) → get relevant API docs
3. Call retriever.search(requirement, domain=detected_domain) → get domain examples

Inject all retrieved context into planning prompt:
"Page structure from crawl:\n{page_context}\n\n
Relevant Playwright APIs:\n{api_context}\n\n
Similar test examples:\n{domain_examples}\n\n
Now create a test plan for: {requirement}"

This makes the planner context-aware of the actual page structure.

---

### B4: main.py — RAG Ingest Mode

Add new --mode ingest to CLI:

uv run python main.py --mode ingest --playwright-docs
uv run python main.py --mode ingest --url https://your-app.com
uv run python main.py --mode ingest --document path/to/spec.pdf
uv run python main.py --mode ingest --all --url https://your-app.com --document spec.pdf

Handler calls RAGIngestionPipeline.ingest_all() with appropriate sources dict.
After ingestion, show summary table.
Also add --mode rag-stats to show current RAG database statistics:
  Total chunks, breakdown by source type, last updated date

---

### B5: pyproject.toml — Add missing dependencies

Add these to dependencies if not already present:
  "httpx>=0.27.0",
  "beautifulsoup4>=4.12.0",
  "markdownify>=0.13.0",
  "pypdf>=5.0.0",
  "python-docx>=1.1.0",
  "rich>=13.9.0",

Run: uv sync after updating pyproject.toml

---

## VERIFICATION SEQUENCE

Run in this exact order:

Step 1 — Test dataset generation (generate 10 examples only as smoke test):
uv run python -c "
import asyncio
from src.data.synthetic_gen import generate_synthetic_dataset
asyncio.run(generate_synthetic_dataset(max_per_domain=1))
print('Dataset generation OK')
"

Step 2 — Test Playwright docs ingestion:
uv run python main.py --mode ingest --playwright-docs

Step 3 — Test URL ingestion:
uv run python main.py --mode ingest --url https://the-internet.herokuapp.com

Step 4 — Test RAG stats:
uv run python main.py --mode rag-stats

Step 5 — Test that generate mode now uses RAG context:
uv run python main.py --mode generate \
  --requirement "test login functionality" \
  --url https://the-internet.herokuapp.com/login \
  --role admin

Check log shows:
  "RAG context: X chunks retrieved from playwright_docs"
  "RAG context: X chunks retrieved from url_crawl"

Step 6 — Full finetune pipeline (only if VRAM is free):
uv run python main.py --mode finetune --step generate
uv run python main.py --mode finetune --step train
uv run python main.py --mode finetune --step export
