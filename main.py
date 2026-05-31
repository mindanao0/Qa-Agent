#!/usr/bin/env python3
"""
Universal Local AI QA Agent — CLI entry point.

Modes:
  generate   Requirement → TestPlan → Playwright script (via agent graph)
  run        Execute an existing test script with pytest + allure
  finetune   QLoRA fine-tune pipeline (generate → train → export)
  heal       Re-validate and re-heal locators in locators.json against a live URL
  ingest     Ingest knowledge into the RAG database (docs, URLs, files)
  rag-stats  Show RAG database statistics

Examples:
  python main.py --mode generate \\
      --requirement "test login functionality" \\
      --url http://localhost:3000 --role admin

  python main.py --mode run --script tests/example_hrm_payroll.py

  python main.py --mode finetune --step all
  python main.py --mode finetune --step generate
  python main.py --mode finetune --step train --dataset ~/.qa-agent/datasets/synthetic_universal.jsonl
  python main.py --mode finetune --step export --checkpoint ~/.qa-agent/checkpoints/final

  python main.py --mode heal --url http://localhost:3000 \\
      --locator-file locators/locators.json

  python main.py --mode ingest --playwright-docs
  python main.py --mode ingest --url https://your-app.com
  python main.py --mode ingest --document path/to/spec.pdf
  python main.py --mode ingest --all --url https://your-app.com --document spec.pdf

  python main.py --mode rag-stats

  python main.py --mode explore --url https://the-internet.herokuapp.com --max-pages 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()

console = Console()

_DEFAULT_DATASET = "~/.qa-agent/datasets/synthetic_universal.jsonl"
_DEFAULT_CHECKPOINT_FINAL = "~/.qa-agent/checkpoints/final"
_AGENT_YAML = Path("config/agent.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa-agent",
        description="Universal Local AI QA Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["generate", "run", "finetune", "heal", "ingest", "rag-stats", "explore"],
        metavar="MODE",
        help="Operation mode",
    )
    # ── shared ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--url",
        default=os.getenv("APP_BASE_URL", "http://localhost:3000"),
        help="Target application URL (default: APP_BASE_URL env)",
    )
    parser.add_argument("--role", default="admin",
                        help="RBAC role for authentication (default: admin)")
    parser.add_argument("--domain", default="general",
                        help="Application domain hint for the planner")
    # ── generate ──────────────────────────────────────────────────────────────
    parser.add_argument("--requirement", default="",
                        help="[generate] Natural-language test requirement")
    parser.add_argument("--output-dir", default="tests/generated",
                        help="[generate] Directory to write the generated script")
    # ── run ───────────────────────────────────────────────────────────────────
    parser.add_argument("--script", default="",
                        help="[run] Path to the test script to execute")
    parser.add_argument("--alluredir", default="allure-results",
                        help="[run] Allure results directory")
    # ── finetune ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--step",
        default="all",
        choices=["generate", "train", "export", "all"],
        help="[finetune] Sub-step to run (default: all)",
    )
    parser.add_argument("--dataset", default=_DEFAULT_DATASET,
                        help="[finetune] Path to JSONL training dataset")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT_FINAL,
                        help="[finetune] Path to the trained checkpoint (for --step export)")
    parser.add_argument("--checkpoint-dir", default="~/.qa-agent/checkpoints",
                        help="[finetune] Directory for saving checkpoints during training")
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="[finetune --step generate] Cap on scenarios per domain "
                             "(smoke-test mode)")
    # ── heal ──────────────────────────────────────────────────────────────────
    parser.add_argument("--locator-file", default="locators/locators.json",
                        help="[heal] Path to locators.json")
    # ── ingest ────────────────────────────────────────────────────────────────
    parser.add_argument("--playwright-docs", action="store_true",
                        help="[ingest] Pull official Playwright Python docs into RAG")
    parser.add_argument("--document", action="append", default=[],
                        help="[ingest] Path to a PDF / DOCX / MD / TXT / JSON / YAML "
                             "document. Repeatable.")
    parser.add_argument("--all", action="store_true",
                        help="[ingest] Run every source given (--playwright-docs, "
                             "--url, --document) in one go")
    # ── explore ───────────────────────────────────────────────────────────────
    parser.add_argument("--max-pages", type=int, default=20,
                        help="[explore] Maximum pages to crawl (default: 20)")
    parser.add_argument("--max-depth", type=int, default=10,
                        help="[explore] Maximum BFS depth (default: 10)")
    # ── global ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M"),
        help="Ollama model name",
    )
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max self-healing retries per test")
    parser.add_argument("--log-level",
                        default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Mode: generate
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_generate(args: argparse.Namespace) -> int:
    if not args.requirement:
        console.print("[red]--requirement is required for generate mode[/red]")
        return 1

    console.print(Panel(
        f"[bold]Generating test script[/bold]\n"
        f"Requirement: {args.requirement}\n"
        f"URL: {args.url}  Role: {args.role}  Domain: {args.domain}",
        title="[green]qa-agent generate[/green]",
    ))

    from src.llm.adapter import OllamaAdapter
    from src.browser.manager import BrowserManager
    from src.agents.graph import build_graph, initial_state
    from src.llm.structured import PlaywrightScript
    from src.rag.retriever import HybridRetriever
    from src.rag.store import VectorStore

    async with OllamaAdapter(model=args.model) as adapter:
        # Best-effort RAG attachment — runs without it if the store is empty.
        retriever: HybridRetriever | None = None
        try:
            vstore = await VectorStore().connect()
            retriever = HybridRetriever(store=vstore, adapter=adapter)
        except Exception as exc:
            logger.warning(f"RAG unavailable, planner will run blind: {exc}")

        async with BrowserManager() as bm:
            graph = build_graph(
                adapter=adapter,
                browser_manager=bm,
                retriever=retriever,
                max_retries=args.max_retries,
            )
            state = initial_state(
                requirement=args.requirement,
                url=args.url,
                role=args.role,
                domain=args.domain,
                mode="generate",
                max_retries=args.max_retries,
            )

            console.print("[dim]Running agent graph (planner → generator)…[/dim]")
            try:
                result = await graph.ainvoke(state)
            except Exception as exc:
                console.print(f"[red]Graph execution failed: {exc}[/red]")
                logger.exception("generate mode failed")
                return 1

            script_dict = result.get("script")
            if not script_dict:
                console.print("[red]No script was generated[/red]")
                return 1

            script = PlaywrightScript.model_validate(script_dict)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fn = script.test_function_name or "test_generated"
            out_path = output_dir / f"{fn}.py"
            out_path.write_text(script.code, encoding="utf-8")

            console.print(f"\n[green]✓ Script saved:[/green] {out_path}")
            console.print(
                Syntax(script.code[:1500], "python", theme="monokai", line_numbers=True)
            )
            if len(script.code) > 1500:
                console.print(f"[dim]… ({len(script.code)} chars total)[/dim]")
            return 0


# ──────────────────────────────────────────────────────────────────────────────
# Mode: run
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_run(args: argparse.Namespace) -> int:
    script_path = Path(args.script) if args.script else None
    if not script_path or not script_path.exists():
        console.print(f"[red]Script not found: {args.script!r}[/red]")
        return 1

    alluredir = Path(args.alluredir)
    alluredir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        str(script_path),
        "-v", "--tb=short", "--no-header",
    ]
    try:
        import importlib.util
        if importlib.util.find_spec("allure") is not None:
            cmd.append(f"--alluredir={alluredir}")
    except Exception:
        pass
    console.print(Panel(
        f"[bold]Executing test script[/bold]\n"
        f"Script: {script_path}\n"
        f"Command: {' '.join(cmd)}",
        title="[green]qa-agent run[/green]",
    ))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        console.print(line.decode(errors="replace").rstrip())
    await proc.wait()

    if proc.returncode == 0:
        console.print("\n[green]✓ All tests passed[/green]")
    else:
        console.print(f"\n[red]✗ Tests failed (exit {proc.returncode})[/red]")
    console.print(
        f"[dim]Allure results: {alluredir}  "
        f"(view with: allure serve {alluredir})[/dim]"
    )
    return proc.returncode


# ──────────────────────────────────────────────────────────────────────────────
# Mode: finetune  (UPDATE 3 — sub-step orchestration)
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_finetune(args: argparse.Namespace) -> int:
    dataset_path = Path(os.path.expanduser(args.dataset))
    checkpoint_dir = Path(os.path.expanduser(args.checkpoint_dir))
    checkpoint_final = Path(os.path.expanduser(args.checkpoint))

    console.print(Panel(
        f"[bold]Fine-tune pipeline[/bold]\n"
        f"Step:        {args.step}\n"
        f"Dataset:     {dataset_path}\n"
        f"Checkpoint:  {checkpoint_final}",
        title="[green]qa-agent finetune[/green]",
    ))

    if args.step in ("generate", "all"):
        rc = await _finetune_generate(dataset_path, args)
        if rc != 0:
            return rc

    if args.step in ("train", "all"):
        rc = await _finetune_train(dataset_path, checkpoint_dir)
        if rc != 0:
            return rc

    if args.step in ("export", "all"):
        # --step export: prefer args.checkpoint; --step all: use the
        # well-known location written by trainer.run()
        target_ckpt = (
            checkpoint_final
            if args.step == "export"
            else (checkpoint_dir / "final")
        )
        rc = await _finetune_export(target_ckpt)
        if rc != 0:
            return rc

    return 0


async def _finetune_generate(
    dataset_path: Path,
    args: argparse.Namespace,
) -> int:
    from src.llm.adapter import OllamaAdapter
    from src.data.synthetic_gen import SyntheticDataGenerator

    full_run = args.max_per_domain is None
    mode_msg = "FULL dataset (all scenarios)" if full_run else f"smoke-test (max {args.max_per_domain}/domain)"
    console.print(
        f"[cyan]Step generate:[/cyan] {mode_msg} → {dataset_path}"
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    async with OllamaAdapter(model=args.model) as adapter:
        gen = SyntheticDataGenerator(adapter=adapter, output_file=dataset_path)
        try:
            count = await gen.generate_dataset(max_per_domain=args.max_per_domain)
        except Exception as exc:
            console.print(f"[red]Dataset generation failed: {exc}[/red]")
            logger.exception("finetune --step generate failed")
            return 1

    console.print(f"[green]✓ Wrote {count} examples → {dataset_path}[/green]")
    _print_dataset_summary(dataset_path)

    if full_run:
        console.print(
            f"\n[bold]Dataset ready at[/bold] {dataset_path}\n"
            "[bold]Next:[/bold] Run WSL2 training with: "
            "[cyan]bash scripts/run_training_wsl2.sh[/cyan]"
        )
    return 0


def _print_dataset_summary(dataset_path: Path) -> None:
    """Read back the JSONL and show a per-domain breakdown table."""
    if not dataset_path.exists():
        return
    from collections import defaultdict
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        with dataset_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("metadata") or {}
                domain = str(meta.get("domain", "?"))
                etype  = str(meta.get("type", "?"))
                counts[domain][etype] += 1
    except OSError as exc:
        logger.warning(f"Could not read dataset for summary: {exc}")
        return

    if not counts:
        return

    table = Table(title="Synthetic dataset breakdown")
    table.add_column("Domain",      style="cyan")
    table.add_column("Happy Path",  justify="right", style="green")
    table.add_column("Edge Cases",  justify="right", style="yellow")
    table.add_column("Healing",     justify="right", style="magenta")
    table.add_column("Other",       justify="right", style="dim")
    table.add_column("Total",       justify="right", style="bold")

    grand = Counter()
    for domain in sorted(counts):
        c = counts[domain]
        happy   = c.get("happy_path", 0)
        edge    = c.get("edge_case", 0)
        healing = c.get("healing", 0)
        other   = sum(v for k, v in c.items() if k not in {"happy_path", "edge_case", "healing"})
        total   = happy + edge + healing + other
        grand["happy"]   += happy
        grand["edge"]    += edge
        grand["healing"] += healing
        grand["other"]   += other
        grand["total"]   += total
        table.add_row(domain, str(happy), str(edge), str(healing), str(other), str(total))

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        str(grand["happy"]),
        str(grand["edge"]),
        str(grand["healing"]),
        str(grand["other"]),
        str(grand["total"]),
    )
    console.print(table)


async def _finetune_train(dataset_path: Path, checkpoint_dir: Path) -> int:
    if not dataset_path.exists():
        console.print(
            f"[red]Dataset not found: {dataset_path}. "
            "Run --step generate first.[/red]"
        )
        return 1

    from src.finetune.trainer import QLoRATrainer

    trainer = QLoRATrainer(checkpoint_dir=str(checkpoint_dir))
    console.print(
        f"[cyan]Step train:[/cyan] dataset={dataset_path} → "
        f"checkpoints={checkpoint_dir}"
    )

    try:
        final_dir = await asyncio.to_thread(trainer.run, dataset_path)
        console.print(f"[green]✓ Training complete. Checkpoint: {final_dir}[/green]")
    except ImportError as exc:
        console.print(f"[red]Missing dependency: {exc}[/red]")
        return 1
    except RuntimeError as exc:
        console.print(f"[red]Training aborted: {exc}[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]Training failed: {exc}[/red]")
        logger.exception("finetune --step train failed")
        return 1
    return 0


async def _finetune_export(checkpoint: Path) -> int:
    from src.finetune.export import ModelExporter

    if not checkpoint.exists():
        console.print(
            f"[red]Checkpoint not found: {checkpoint}. "
            "Run --step train first.[/red]"
        )
        return 1

    exporter = ModelExporter(checkpoint_dir=str(checkpoint))
    try:
        gguf_path = await asyncio.to_thread(exporter.merge_and_export)
        console.print(f"[green]✓ Exported GGUF: {gguf_path}[/green]")
    except ImportError as exc:
        console.print(f"[red]Missing dependency: {exc}[/red]")
        return 1
    except Exception as exc:
        console.print(f"[red]GGUF export failed: {exc}[/red]")
        logger.exception("finetune --step export failed")
        return 1

    # Update config/agent.yaml to point at the newly-registered model
    try:
        _update_agent_yaml_model("qa-agent-coder")
        console.print(
            "[green]✓ config/agent.yaml llm.model updated → qa-agent-coder[/green]"
        )
    except Exception as exc:
        logger.warning(f"Could not update config/agent.yaml: {exc}")
        console.print(
            f"[yellow]⚠ Edit config/agent.yaml manually to set "
            f"llm.model: qa-agent-coder ({exc})[/yellow]"
        )
    return 0


def _update_agent_yaml_model(new_model: str) -> None:
    """
    Rewrite the `model:` line under the `llm:` block of config/agent.yaml.
    Plain-text edit (no yaml round-trip) preserves comments and ordering.
    """
    if not _AGENT_YAML.exists():
        raise FileNotFoundError(_AGENT_YAML)

    text = _AGENT_YAML.read_text(encoding="utf-8")
    # Only rewrite the first model: line that sits inside the llm: block.
    new_text = re.sub(
        r"(?m)^(\s*model:\s*)\".*\"\s*$",
        rf'\1"{new_model}"',
        text,
        count=1,
    )
    if new_text == text:
        # No quoted form found — try unquoted as a fallback
        new_text = re.sub(
            r"(?m)^(\s*model:\s*)\S+\s*$",
            rf'\1"{new_model}"',
            text,
            count=1,
        )
    if new_text == text:
        raise RuntimeError("Could not locate `model:` key in config/agent.yaml")
    _AGENT_YAML.write_text(new_text, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: heal
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_heal(args: argparse.Namespace) -> int:
    locator_file = Path(args.locator_file)
    if not locator_file.exists():
        console.print(
            f"[yellow]Locator file not found: {locator_file} — nothing to heal[/yellow]"
        )
        return 0

    locators: dict[str, Any] = json.loads(locator_file.read_text())
    if not locators:
        console.print("[yellow]Locator file is empty — nothing to heal[/yellow]")
        return 0

    console.print(Panel(
        f"[bold]Re-validating locators[/bold]\n"
        f"Locator file: {locator_file}\n"
        f"Base URL: {args.url}\n"
        f"Entries: {len(locators)}",
        title="[green]qa-agent heal[/green]",
    ))

    from src.llm.adapter import OllamaAdapter
    from src.browser.manager import BrowserManager
    from src.browser.auth_manager import AuthManager
    from src.healing.engine import HealingEngine

    healed_count = 0
    failed_count = 0

    async with OllamaAdapter(model=args.model) as adapter:
        engine = HealingEngine(adapter=adapter, locator_file=str(locator_file))
        async with BrowserManager() as bm:
            try:
                auth_mgr = AuthManager()
            except Exception:
                auth_mgr = None

            for locator_str, entry in locators.items():
                url = entry.get("url_pattern") or args.url
                role = args.role

                console.print(f"  Healing [cyan]{locator_str[:60]}[/cyan]…", end=" ")
                try:
                    storage_state = (
                        await auth_mgr.get_storage_state(role) if auth_mgr else None
                    )
                    async with bm.new_page(storage_state=storage_state) as page:
                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                        except Exception:
                            pass
                        healed = await engine.heal(
                            page=page,
                            failed_locator=locator_str,
                            action="validate",
                            error_message="Scheduled re-heal from CLI",
                        )
                    if healed.confidence >= 0.85:
                        console.print(
                            f"[green]✓ confidence={healed.confidence:.2f} "
                            f"method={healed.method}[/green]"
                        )
                        healed_count += 1
                    else:
                        console.print(
                            f"[yellow]⚠ low confidence={healed.confidence:.2f}[/yellow]"
                        )
                        failed_count += 1
                except Exception as exc:
                    console.print(f"[red]✗ {exc}[/red]")
                    failed_count += 1

    _print_heal_summary(healed_count, failed_count)
    return 0 if failed_count == 0 else 1


def _print_heal_summary(healed: int, failed: int) -> None:
    table = Table(title="Heal Summary")
    table.add_column("Status", style="bold")
    table.add_column("Count")
    table.add_row("[green]Healed (≥0.85)[/green]", str(healed))
    table.add_row("[red]Low confidence / failed[/red]", str(failed))
    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: ingest  (UPDATE 3 — RAG ingestion)
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_ingest(args: argparse.Namespace) -> int:
    """
    Run any combination of the three RAG sources.

      --playwright-docs            ingest curated Playwright Python docs
      --url URL [--url URL ...]    crawl one or more target apps
      --document PATH [--doc ...]  ingest PDF / DOCX / MD / TXT / JSON / YAML
      --all                        run every flag given in a single batch
    """
    from src.llm.adapter import OllamaAdapter
    from src.rag.ingestion import RAGIngestionPipeline
    from src.rag.store import VectorStore

    urls: list[str] = []
    if args.url and args.url != "http://localhost:3000":
        urls.append(args.url)

    documents: list[str] = list(args.document or [])
    wants_playwright = bool(args.playwright_docs)

    if not (wants_playwright or urls or documents):
        console.print(
            "[red]Specify at least one source: --playwright-docs, --url, --document[/red]"
        )
        return 1

    console.print(Panel(
        f"[bold]Ingesting into RAG[/bold]\n"
        f"playwright_docs: {wants_playwright}\n"
        f"urls:            {urls or '—'}\n"
        f"documents:       {documents or '—'}",
        title="[green]qa-agent ingest[/green]",
    ))

    async with OllamaAdapter(model=args.model) as adapter:
        vstore = await VectorStore().connect()
        pipeline = RAGIngestionPipeline(store=vstore, adapter=adapter)

        try:
            sources_dict: dict[str, Any] = {
                "playwright_docs": wants_playwright,
                "urls": urls,
                "documents": documents,
            }
            results = await pipeline.ingest_all(sources_dict, console=console)
        except Exception as exc:
            console.print(f"[red]Ingestion failed: {exc}[/red]")
            logger.exception("ingest mode failed")
            return 1

    total = sum(results.values())
    console.print(f"[green]✓ Ingestion complete — {total} chunks added[/green]")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Mode: rag-stats
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_rag_stats(args: argparse.Namespace) -> int:
    """Print a summary of the current RAG database (chunks, sources, freshness)."""
    from src.rag.store import VectorStore

    try:
        vstore = await VectorStore().connect()
    except Exception as exc:
        console.print(f"[red]Could not open vector store: {exc}[/red]")
        return 1

    try:
        total = await vstore.count()
        all_chunks = await vstore.get_all_chunks()
    except Exception as exc:
        console.print(f"[red]Failed to read vector store: {exc}[/red]")
        return 1

    by_type: Counter[str] = Counter(c.doc_type for c in all_chunks)
    sources: Counter[str] = Counter(c.source for c in all_chunks)
    last_updated = max((c.created_at for c in all_chunks if c.created_at), default="—")

    table = Table(title="RAG Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total chunks", str(total))
    table.add_row("Distinct sources", str(len(sources)))
    table.add_row("Last updated", last_updated)
    console.print(table)

    if by_type:
        type_table = Table(title="Breakdown by doc_type")
        type_table.add_column("doc_type", style="cyan")
        type_table.add_column("Chunks", justify="right", style="green")
        for doc_type, count in by_type.most_common():
            type_table.add_row(doc_type, str(count))
        console.print(type_table)

    if sources:
        top_sources = sources.most_common(10)
        src_table = Table(title="Top 10 sources by chunk count")
        src_table.add_column("Source", style="cyan", no_wrap=False)
        src_table.add_column("Chunks", justify="right", style="green")
        for source, count in top_sources:
            label = source if len(source) <= 80 else source[:77] + "…"
            src_table.add_row(label, str(count))
        console.print(src_table)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Mode: explore  (Sprint 4 — SFG crawler integration)
# ──────────────────────────────────────────────────────────────────────────────


async def _mode_explore(args: argparse.Namespace) -> int:
    """BFS-crawl a URL and build the State Flow Graph (SFG) in the local DB."""
    from src.contractskill.sfg import SFGStore
    from src.contractskill.crawler import SFGCrawler, CrawlerConfig
    from src.perception.grounder import Grounder

    if not args.url or args.url == "http://localhost:3000":
        console.print("[red]--url is required for explore mode[/red]")
        return 1

    console.print(Panel(
        f"[bold]Exploring State Flow Graph[/bold]\n"
        f"URL:       {args.url}\n"
        f"max-pages: {args.max_pages}\n"
        f"max-depth: {args.max_depth}",
        title="[green]qa-agent explore[/green]",
    ))

    sfg_store = SFGStore()
    grounder = Grounder()
    config = CrawlerConfig(max_pages=args.max_pages, max_depth=args.max_depth)
    crawler = SFGCrawler(sfg_store, grounder, config)

    console.print("[dim]Starting BFS crawl…[/dim]")
    try:
        await crawler.crawl(args.url)
    except Exception as exc:
        console.print(f"[red]Crawl failed: {exc}[/red]")
        logger.exception("explore mode failed")
        return 1

    nodes = sfg_store.node_count()
    edges = sfg_store.edge_count()
    console.print(
        f"[green]Crawl complete.[/green] "
        f"Nodes: {nodes}, Edges: {edges}"
    )
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    _handlers = {
        "generate":  _mode_generate,
        "run":       _mode_run,
        "finetune":  _mode_finetune,
        "heal":      _mode_heal,
        "ingest":    _mode_ingest,
        "rag-stats": _mode_rag_stats,
        "explore":   _mode_explore,
    }

    exit_code: int = asyncio.run(_handlers[args.mode](args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
