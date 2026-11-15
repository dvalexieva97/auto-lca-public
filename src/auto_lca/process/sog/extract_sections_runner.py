"""
Offline (non-AI) rebuild of extracted_sections/{pid}.json beside existing extracts.
"""

from __future__ import annotations

from pathlib import Path

from auto_lca.process.nlp.extract import TextExtractor
from auto_lca.process.sog.section_context import (
    build_sections_dict,
    save_extracted_sections,
)


def run_extract_sections(args) -> None:
    extracts_dir = Path(args.extracts_dir)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else extracts_dir.parent / "input-papers"
    paths = sorted(extracts_dir.glob("*.json"))
    paths = [p for p in paths if p.name != "summary.json"]
    if args.limit is not None:
        paths = paths[: args.limit]

    extractor = TextExtractor()
    n_ok = n_skip = n_fail = 0
    print(f"extract sections → {extracts_dir / 'extracted_sections'}")
    for i, path in enumerate(paths, 1):
        pid = path.stem
        out = extracts_dir / "extracted_sections" / f"{pid}.json"
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        pdf = pdf_dir / f"{pid}.pdf"
        if not pdf.is_file():
            # try metadata.pdf_path
            try:
                import json

                meta = json.loads(path.read_text(encoding="utf-8")).get("metadata") or {}
                raw = meta.get("pdf_path")
                if raw and Path(raw).is_file():
                    pdf = Path(raw)
            except Exception:  # noqa: BLE001
                pass
        if not pdf.is_file():
            print(f"[{i}/{len(paths)}] {pid[:12]}… ❌ no PDF")
            n_fail += 1
            continue
        try:
            text = extractor.extract_text_from_pdf(str(pdf)) or ""
            sections = build_sections_dict(text)
            save_extracted_sections(extracts_dir, pid, sections)
            keys = [k for k in sections if k != "full_text"]
            print(f"[{i}/{len(paths)}] {pid[:12]}… ok sections={keys}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(paths)}] {pid[:12]}… ❌ {e}")
            n_fail += 1
    print(f"done: wrote={n_ok} skipped={n_skip} failed={n_fail}")
