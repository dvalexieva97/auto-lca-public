"""
Batch PDF extraction via AgentExtractor (writes per-paper {pid}.json).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from auto_lca.process.sog.agent_sog import AgentExtractor
from auto_lca.process.sog.results_table import write_extract_results_xlsx
from auto_lca.process.sog.section_context import save_extracted_sections


def expand_pdf_inputs(paths: list[str]) -> list[str]:
    """Expand ``--pdfs`` values: each entry may be a PDF file or a folder of PDFs."""
    out: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(
                p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
            )
            if not found:
                raise SystemExit(f"No PDF files found in folder: {path}")
            out.extend(str(p) for p in found)
        elif path.is_file():
            if path.suffix.lower() != ".pdf":
                raise SystemExit(f"Not a PDF file: {path}")
            out.append(str(path))
        else:
            raise SystemExit(
                f"Not a PDF file or folder: {path}\n"
                "Pass one or more .pdf paths, or a folder that contains PDFs."
            )
    return out


def run_extract(args):
    # A fresh, timestamp-salted namespace by default (see AgentExtractor's
    # cache_namespace) -- guarantees this run's prompt_cache_key values
    # were never touched by an earlier process (e.g. ad-hoc testing calls
    # against real paper PIDs), so cache-hit numbers reported below
    # reflect this run alone, not leftover warm entries. Mistral's prompt
    # cache has no manual clear/purge API; a fresh namespace is the only
    # way to guarantee a cold start rather than trusting a TTL already
    # expired.
    extractor = AgentExtractor(
        config_path=args.config,
        model=args.model,
        scenario_model=args.scenario_model,
        model_profile=getattr(args, "model_profile", None),
        domain=getattr(args, "domain", None),
        entry_default_domain=getattr(args, "entry_default_domain", "lca"),
    )
    print(
        f"cache namespace for this run: {extractor.cache_namespace!r} (fresh -- no prior calls can hit it)"
    )
    print(
        f"domain={extractor.domain}  models: extraction={extractor.model}  "
        f"scenarios={extractor.scenario_model}  (profile={extractor.model_profile})"
    )
    os.makedirs(args.output, exist_ok=True)

    pdfs = expand_pdf_inputs(list(args.pdfs))
    print(f"papers: {len(pdfs)} PDF(s) to process")

    run_prompt_tokens = 0
    run_cached_tokens = 0
    n_done = 0
    n_skipped = 0
    n_failed = 0
    for pdf_path in pdfs:
        pid = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(args.output, f"{pid}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(
                f"skip {pid}: already processed "
                f"({out_path} exists; pass --overwrite to re-extract)"
            )
            n_skipped += 1
            continue
        print(f"extracting {pid}...")
        t0 = time.time()
        try:
            data = extractor.full_extract(pdf_path, scenarios_only=args.scenarios_only)
        except Exception as e:  # noqa: BLE001 -- one paper's failure (e.g. a
            # rate-limit window that outlasts all retries) must not lose
            # progress on the rest of the batch. Confirmed live: an
            # uncaught RuntimeError from full_extract_async's scenario-ID
            # call crashed the whole run_extract loop, discarding every
            # already-written paper's *result* only from the CALLER's
            # perspective (the .json files on disk were fine) but forcing
            # a full manual re-invocation -- the skip-existing check above
            # already makes that safe/resumable, so just log and continue.
            print(f"  ❌ failed after retries, skipping this paper: {e}")
            n_failed += 1
            continue
        data["metadata"]["exec_time"] = time.time() - t0
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        sections = data.get("sections_dict")
        if isinstance(sections, dict) and sections:
            save_extracted_sections(args.output, pid, sections)
        tokens = data.get("tokens", {})
        usage = tokens.get("field_tokens", tokens.get("aspect_tokens", {}))
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        cached_tokens = usage.get("cached_tokens", 0) or 0
        run_prompt_tokens += prompt_tokens
        run_cached_tokens += cached_tokens
        n_done += 1
        cache_pct = (
            f"{100 * cached_tokens / prompt_tokens:.1f}%" if prompt_tokens else "n/a"
        )
        print(
            f"  -> {len(data['results'])} scenario groups, {time.time() - t0:.0f}s, "
            f"cache hit {cache_pct} of prompt tokens"
        )

    if n_done:
        overall_pct = (
            f"{100 * run_cached_tokens / run_prompt_tokens:.1f}%"
            if run_prompt_tokens
            else "n/a"
        )
        print(
            f"\nRun cache summary: {run_cached_tokens:,}/{run_prompt_tokens:,} prompt tokens "
            f"were cache hits ({overall_pct}) across {n_done} paper(s) extracted this run."
        )
    if n_skipped or n_failed:
        print(
            f"Run tally: {n_done} extracted, {n_skipped} skipped (already processed), "
            f"{n_failed} failed."
        )

    # One spreadsheet for the whole output folder (includes prior extracts too).
    try:
        xlsx_path = write_extract_results_xlsx(args.output)
    except Exception as e:  # noqa: BLE001 -- never fail the run on export
        print(f"⚠️  could not write extracted_results.xlsx: {e}")
    else:
        if xlsx_path:
            print(f"Joined extract table: {xlsx_path}")
