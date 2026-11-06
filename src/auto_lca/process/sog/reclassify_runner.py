"""
In-place scenario_category refresh on existing agent extract JSONs.

Re-runs only ``_filter_valid_scenarios_async`` (one LLM call per paper):
candidate names/descriptions + Results/Tables/Methods section context
(from extracted_sections/{pid}.json when present, else rebuilt from the PDF).
Does not re-identify scenarios or re-extract concept fields.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from auto_lca.process.sog.agent_sog import AgentExtractor
from auto_lca.process.sog.section_context import (
    build_sections_dict,
    load_extracted_sections,
    save_extracted_sections,
    validator_context_from_sections,
)


def _scenario_block(group: list) -> dict | None:
    if not group or not isinstance(group[0], dict):
        return None
    block = group[0].get("Scenario")
    return block if isinstance(block, dict) else None


def _resolve_pdf(
    data: dict, extracts_dir: Path, json_path: Path, pdf_dir: Path | None
) -> Path | None:
    meta = data.get("metadata") or {}
    raw = meta.get("pdf_path")
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else Path.cwd() / p)
    pid = str(meta.get("pid") or json_path.stem)
    if pdf_dir:
        candidates.append(Path(pdf_dir) / f"{pid}.pdf")
    candidates.append(extracts_dir.parent / "input-papers" / f"{pid}.pdf")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _paper_context(
    extractor: AgentExtractor,
    extracts_dir: Path,
    pid: str,
    pdf_path: Path | None,
    *,
    refresh_sections: bool,
) -> str:
    sections = load_extracted_sections(extracts_dir, pid)
    if sections is None or refresh_sections:
        if pdf_path is None:
            return ""
        try:
            text = extractor.extract_text_from_pdf(str(pdf_path)) or ""
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ PDF text failed ({pdf_path.name}): {e}")
            return ""
        sections = build_sections_dict(text)
        save_extracted_sections(extracts_dir, pid, sections)
    return validator_context_from_sections(
        sections,
        max_chars=extractor.VALIDATOR_CONTEXT_CHARS,
        intro_chars=extractor.PAPER_CONTEXT_CHARS,
    )


async def _reclassify_one(
    extractor: AgentExtractor,
    path: Path,
    pdf_dir: Path | None,
    *,
    dry_run: bool,
    refresh_sections: bool,
) -> list[tuple[str, str, str]]:
    """Return list of (name, old_cat, new_cat) for changed rows."""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    candidates = []
    blocks: list[dict] = []
    for group in results:
        block = _scenario_block(group)
        if not block:
            continue
        name = block.get("Scenario Name") or block.get("name") or ""
        desc = block.get("scenario_description") or block.get("description") or ""
        candidates.append({"name": name, "description": desc})
        blocks.append(block)

    if not candidates:
        return []

    pid = str((data.get("metadata") or {}).get("pid") or path.stem)
    pdf_path = _resolve_pdf(data, path.parent, path, pdf_dir)
    context = _paper_context(
        extractor,
        path.parent,
        pid,
        pdf_path,
        refresh_sections=refresh_sections,
    )
    classified, _usage = await extractor._filter_valid_scenarios_async(
        candidates, paper_context=context, preserve_all=True
    )
    by_name = {c.get("name"): c for c in classified}

    changes: list[tuple[str, str, str]] = []
    for block in blocks:
        name = block.get("Scenario Name") or block.get("name") or ""
        old = str(block.get("scenario_category") or "").strip()
        tagged = by_name.get(name) or {}
        new = str(
            tagged.get("scenario_category")
            or extractor.SCENARIO_CATEGORY_UNCLASSIFIED
        ).strip()
        def_only = new in extractor._definition_only_categories()
        if old != new or bool(block.get("definition_only")) != def_only:
            changes.append((name, old or "∅", new))
        block["scenario_category"] = new
        block["definition_only"] = def_only

    if changes and not dry_run:
        meta = data.setdefault("metadata", {})
        meta["scenario_categories_reclassified"] = True
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changes


async def _reclassify_folder_async(
    extractor: AgentExtractor,
    extracts_dir: Path,
    pdf_dir: Path | None,
    *,
    dry_run: bool,
    limit: int | None,
    refresh_sections: bool,
) -> Counter:
    paths = sorted(extracts_dir.glob("*.json"))
    paths = [p for p in paths if p.name != "summary.json"]
    if limit is not None:
        paths = paths[:limit]

    totals: Counter = Counter()
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.stem[:12]}…")
        try:
            changes = await _reclassify_one(
                extractor,
                path,
                pdf_dir,
                dry_run=dry_run,
                refresh_sections=refresh_sections,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {e}")
            totals["errors"] += 1
            continue
        if not changes:
            print("  (no category changes)")
            totals["unchanged_papers"] += 1
            continue
        totals["changed_papers"] += 1
        for name, old, new in changes:
            totals[f"{old}→{new}"] += 1
            print(f"  {name[:60]!r}: {old} → {new}")
            if new == extractor.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED:
                totals["→mentioned"] += 1
            if new == extractor.SCENARIO_CATEGORY_OWN_SYSTEM_MODELLED:
                totals["→modelled"] += 1
    return totals


def run_reclassify_scenarios(args) -> None:
    extracts_dir = Path(args.extracts_dir)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
    extractor = AgentExtractor(
        config_path=args.config,
        model=args.model,
        scenario_model=args.scenario_model,
        model_profile=getattr(args, "model_profile", None),
    )
    print(
        f"reclassify scenarios in {extracts_dir}\n"
        f"  config={args.config}\n"
        f"  scenario_model={extractor.scenario_model}\n"
        f"  context=Results/Tables/Methods sections"
        f" (max {extractor.VALIDATOR_CONTEXT_CHARS} chars)"
        f"{' (dry-run)' if args.dry_run else ''}"
    )
    totals = asyncio.run(
        _reclassify_folder_async(
            extractor,
            extracts_dir,
            pdf_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            refresh_sections=bool(getattr(args, "refresh_sections", False)),
        )
    )
    print("\nSummary:")
    for k, v in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
