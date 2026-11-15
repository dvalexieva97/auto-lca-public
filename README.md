# AUTO-LCA

**AUTO-LCA** extracts structured data from Life Cycle Assessment (LCA) studies. You describe the
data you want in a configuration file (field sets and fields), and AUTO-LCA reads the PDFs with
Mistral AI's structured output generation and returns one record per paper and product system.
It can also compare the extracted data against a manually compiled reference dataset.

## What It Does

- **Extraction** (`auto-lca extract` / `auto-extract extract`): identifies the product systems
  (or cases) each paper models and extracts the configured fields, once per paper or once per
  system. `auto-lca` defaults to the **LCA** prompt pack; `auto-extract` defaults to the
  **generic** (non-LCA) pack. Same pipeline otherwise.
- **Comparison** (`auto-lca compare`): matches the extracted product systems to those of a
  reference dataset and scores the agreement field by field (cosine, fuzzy and numeric
  similarity, and optionally an LLM judge).

## Release notes

### November 2026 — Large 4 removed from free defaults

Mistral Large 4 (`mistral-large-2512` / Large 4 class) was removed from the **free-tier**
default model pair. Free accounts now default to Ministral only
(`ministral-14b-2512` for field extraction and product-system identification). Paid /
EDU defaults are unchanged (`mistral-small-2603` + `mistral-large-2512`). Override with
config or `--model` / `--scenario-model` when your plan allows it.

### October 2026 — Mistral free-tier model defaults

From October 2026, Mistral changed what is usable on a free API account: models that
previously worked on the free tier (notably Mistral Small and Large snapshots, and more
recently Mistral Large 4 on some free orgs) are often blocked or rate-limited to zero,
while Ministral remains usable. See
[Mistral AI free tier limits](https://freellmhub.com/en/providers/mistral-free/) for a
summary of the current free-tier picture.

AUTO-LCA therefore picks default models from your Mistral plan
(`GET /v1/billing/subscription` → `plan_name`):

| Plan | Field extraction / scenario mapper | Product-system identification |
|--|--|--|
| Free (`plan_name=FREE`) | `ministral-14b-2512` | `ministral-14b-2512` |
| Paid / EDU / other non-free | `mistral-small-2603` | `mistral-large-2512` |

You can still override models in the config or with `--model` / `--scenario-model`. If an
override is not allowed for your plan, extract fails early and suggests the plan defaults
above.

## Installation

You need **Python 3.11 or newer**
([install Python](https://www.python.org/downloads/) — use the official installer for
Windows/macOS, or your package manager on Linux) and a
[Mistral API key](https://console.mistral.ai). The same steps work on macOS, Linux and
Windows. CI tests **3.11, 3.13, 3.14, and 3.15**; **3.13** is the version we develop
against day to day.

```bash
pip install "git+https://github.com/dvalexieva97/auto-lca-public.git"
```

Or, with [uv](https://docs.astral.sh/uv/), as a standalone command:

```bash
uv tool install "git+https://github.com/dvalexieva97/auto-lca-public.git"
```

The first install takes a few minutes: it pulls in PyTorch (about 1 GB), used for optional
section splitting and for field similarity scoring in `compare`.

Optional extras (add them in brackets, e.g. `pip install "auto-lca[eval] @ git+https://github.com/dvalexieva97/auto-lca-public.git"`):

| Extra | Adds |
|--|--|
| `eval` | BERTScore sheet in `auto-lca compare` |
| `tables` | Docling / Tabula table extraction (`auto_lca.process.nlp.tables`; Tabula needs Java) |
| `supplementary` | Read `.docx` / legacy `.xls` side materials alongside PDFs |
| `all` | All of the above |

### API keys

Set them as environment variables, or in a `.env` file in the folder you run `auto-lca` from:

```bash
MISTRAL_API_KEY="your-mistral-key"     # required: extraction and product system matching
OPENAI_API_KEY="your-openai-key"       # optional: only for `auto-lca compare --llm-judge`
```

### Try it in a clean environment

To test the package without touching your other Python setups, install it into a new virtual
environment in an empty folder.

You also need an LCA paper as a PDF. Any will do; a short open-access example is
Baldini et al. (2020), *Environmental Sustainability Assessment of Dairy Farms Rearing the
Italian Simmental Dual-Purpose Breed* (13 pages, CC BY 4.0):
<https://doi.org/10.3390/ani10020296>. Open the link, click **Download PDF**, and save it in
the folder below as `baldini2020.pdf`.

```bash
mkdir auto-lca-test && cd auto-lca-test
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "git+https://github.com/dvalexieva97/auto-lca-public.git"

auto-lca --version                 # the command is installed
echo 'MISTRAL_API_KEY="your-mistral-key"' > .env

auto-lca sample-config --domain lca --format xlsx   # writes sample_config_lca.xlsx
auto-lca extract --pdfs baldini2020.pdf --config sample_config_lca.xlsx --output results
```

`results/baldini2020.json` holds the product systems and fields;
`results/extracted_results.xlsx` is a one-row-per-scenario table of every paper in
`results/`. If you skip `sample-config`, `extract` can copy a bundled `sample_config_*`
file into the folder when you pass that exact filename as `--config`.

## Quick start

```bash
auto-lca sample-config --domain lca --format xlsx   # Pipeline + Field sets sheets
auto-lca extract --pdfs papers/ --config sample_config_lca.xlsx --output results
```

## Generic (non-LCA) extraction

The extract pipeline is shared. Only the **prompt pack** changes:

| Entry | Default domain |
|--|--|
| `auto-lca …` | `lca` |
| `auto-extract …` | `generic` |

Precedence for the domain: `--domain` on the CLI → `defaults.domain` in the config →
entry-point default above. The chosen domain is written into each result JSON as
`metadata.domain`.

`auto-extract` is registered next to `auto-lca`. After install (or `pip install -e .` from a
clone), both commands should print a version:

```bash
auto-lca --version
auto-extract --version
```

If `auto-extract` is missing, reinstall the package, or force the generic pack on the LCA entry:

```bash
auto-lca extract --domain generic --pdfs paper.pdf --config sample_config_generic.json --output out
```

### Example

```bash
auto-extract sample-config --domain generic --format json

auto-extract extract \
  --pdfs paper.pdf \
  --config sample_config_generic.json \
  --output out
```

`sample_config_generic.json` sets `"domain": "generic"`, turns off scenario identification,
keeps models as a nested `Models` list, and enables PDF-table injection +
`allow_additional_fields`. The matching **XLSX** sample is different on purpose:
scenarios on, flat Name/Score fields with `by_production_system=true` (spreadsheets
cannot express list-of-objects).

## Configuration

Two bundled starters (JSON + XLSX each):

| Sample | Command |
|--|--|
| LCA | `auto-lca sample-config --domain lca --format json\|xlsx` |
| Generic | `auto-extract sample-config --domain generic --format json\|xlsx` |

XLSX has two sheets:

- **Pipeline** — every pipeline knob (`key`, `value`, `description`) with domain-appropriate defaults
- **Field sets** — one row per field (`field_set`, `description`, `by_production_system`, `field`, `data_type`)

**XLSX cannot express nested field types.** Spreadsheet `data_type` cells are scalar
strings only (`str`, `float | None`, …). Nested objects
(`{"amount": "float | None", "unit": "str | None"}`) and lists of objects
(`[{"Name": "str | None", "Score": "float | None"}]`) require the **JSON** config.
The LCA JSON sample uses nested `amount`/`unit` impacts (Carbon footprint, Acidification,
Eutrophication). The LCA XLSX uses the same field names with `str | None` (one cell for the value as
stated in the paper — number and unit together).
The generic JSON sample uses a paper-level `Models` list (scenarios off); the generic XLSX
turns scenarios on and uses flat Name/Score fields per identified case.

### JSON format

```json
{
  "defaults": {
    "domain": "lca",
    "extract_production_systems": true,
    "extract_non_primary_production_systems": false,
    "by_production_system": true,
    "strip_bibliography": true,
    "extract_sections": false,
    "include_pdf_tables": false,
    "allow_additional_fields": false
  },
  "field_sets": [
    {
      "name": "Lifecycle Assessment Characteristics",
      "description": "Core LCA framework characteristics...",
      "by_production_system": false,
      "fields": {
        "Functional Unit": "str",
        "System Boundary": "str"
      }
    }
  ]
}
```

| Setting | LCA sample | Generic sample | Meaning |
|--|--|--|--|
| `domain` | `lca` | `generic` | Prompt pack |
| `model` / `production_system_model` / `scenario_mapper_model` | *(omit = plan default)* | *(omit)* | Optional overrides |
| `extract_production_systems` | `true` | `false` | Identify product systems / scenarios |
| `extract_non_primary_production_systems` | `false` | `false` | Extract fields for non-primary systems |
| `by_production_system` | `true` | `false` | Default grain when a field set omits the flag |
| `strip_bibliography` | `true` | `true` | Drop the reference list |
| `extract_sections` | `false` | `false` | Write `extracted_sections/` |
| `include_pdf_tables` | `false` | `true` | Append structured PDF table grids |
| `allow_additional_fields` | `false` | `true` | Keep extra keys the model adds |

Older configs that use `concepts` / `structure`, or `scenario` instead of `production_system`
in the keys, still load.

## Extract data from papers

```bash
auto-lca extract --pdfs papers/ --config sample_config_lca.xlsx --output results
```

| Option | Meaning |
|--|--|
| `--pdfs` | One or more PDF files and/or folders of PDFs |
| `--config` | Configuration (`.json` or `.xlsx`) |
| `--output` | Folder for results |
| `--domain` | `lca` or `generic` prompt pack (overrides config / entry default) |
| `--model` / `--scenario-model` | Override plan / config models (checked against your Mistral plan) |
| `--model-profile` | `auto` (default), `paid`, or `free`: force which plan default pair to use |
| `--scenarios-only` | Identify product systems only, without field extraction |
| `--overwrite` | Re-extract papers that already have a result |

Papers that already have a result in `--output` are skipped, so an interrupted run can be
resumed by running the same command again.

**Output:**

- `results/<paper id>.json`: one file per PDF (the paper ID is the PDF file name), with the
  product systems, the extracted fields, token usage and run metadata.
- `results/extracted_results.xlsx`: joined table of every extract in the folder (one row per
  scenario). Nested objects become nested header levels; list fields are one JSON cell.
- `results/extracted_sections/`: only when `defaults.extract_sections` is `true` — detected
  paper sections (Methods, Results, …) per PDF.

## Compare against a reference dataset

```bash
auto-lca compare --output-folder results --manual-xlsx reference.xlsx --manual-sheet Sheet1
```

`--output-folder` is the `--output` folder of `auto-lca extract`. The reference workbook has one
row per paper and product system, and two header rows (the second and third rows of the sheet):
the field set, then the field, matching the names in your configuration. It also needs:

| Field set | Field | Content |
|--|--|--|
| `Metadata` | `PID` | Paper ID: the PDF file name without `.pdf` |
| `Study details` | `Scenario Name` | Name of the product system |
| `Metadata` | `Downloaded` | Optional: `No` for papers you could not obtain |

The results are written to `results/stats/joined_*.xlsx`: extracted and reference values side by
side, similarity per field, the product system matching, and summary statistics.

| Option | Meaning |
|--|--|
| `--llm-judge` | Also judge with an LLM whether each extracted value means the same as the reference (needs `OPENAI_API_KEY`) |
| `--excluded-fields` | JSON file with a list of `{field, source, reason}`: fields left out of the scoring |
| `--mapping-only` | Only match product systems, without scoring |

See `auto-lca compare --help` for all options.

## Other commands

| Command | What it does |
|--|--|
| `sample-config` | Copy a bundled example configuration (`lca` or `generic`, `json` or `xlsx`) |
| `reclassify-scenarios` | Re-label product system categories on existing extracts |
| `extract-sections` | Rebuild `extracted_sections/` for existing extracts |
| `enrich-excel` | Verify missing reference fields against the PDFs and fill them in |
| `regenerate-stats` | Recompute statistics from an existing joined workbook |

Run `auto-lca <command> --help` (or `auto-extract <command> --help`) for the options.
`python -m auto_lca …` works the same as `auto-lca …` (LCA defaults).

## Developing and running the tests

```bash
cd /path/to/auto-lca
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
auto-lca --version
auto-extract --version
pytest
```

The tests need no API keys and make no API calls.

## Troubleshooting

**"MISTRAL_API_KEY environment variable not set" or authentication errors:** set
`MISTRAL_API_KEY` in your environment, or in a `.env` file in the folder you run `auto-lca` from.

**Models not usable / tier errors on extract:** your key's Mistral plan cannot call the
models in the config (or the CLI overrides). Remove those overrides so plan defaults apply,
or switch to a paid plan. See [Release notes](#october-2026--mistral-free-tier-model-defaults).

**"Config file not found":** check the path to your config and that it ends in `.json` or
`.xlsx`.

**A PDF fails:** check that it is not corrupted and contains text (not only scanned images). A
paper that fails is logged and skipped; rerun the same command to retry only the missing ones.
