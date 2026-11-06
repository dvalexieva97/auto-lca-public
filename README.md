# AUTO-LCA

**AUTO-LCA** extracts structured data from Life Cycle Assessment (LCA) studies. You describe the
data you want in a configuration file (field sets and fields), and AUTO-LCA reads the PDFs with
Mistral AI's structured output generation and returns one record per paper and product system.
It can also compare the extracted data against a manually compiled reference dataset.

## What It Does

- **Extraction** (`auto-lca extract`): identifies the product systems each paper models and
  extracts the configured fields, once per paper or once per product system.
- **Comparison** (`auto-lca compare`): matches the extracted product systems to those of a
  reference dataset and scores the agreement field by field (cosine, fuzzy and numeric
  similarity, and optionally an LLM judge).

## Release notes

### October 2026 — Mistral free-tier model defaults

From October 2026, Mistral changed what is usable on a free API account: models that
previously worked on the free tier (notably Mistral Small and older Large snapshots) are
often blocked or rate-limited to zero for free organizations, while Ministral and
Mistral Large 4 remain usable. See
[Mistral AI free tier limits](https://freellmhub.com/en/providers/mistral-free/) for a
summary of the current free-tier picture.

AUTO-LCA therefore picks default models from your Mistral plan
(`GET /v1/billing/subscription` → `plan_name`):

| Plan | Field extraction / scenario mapper | Product-system identification |
|--|--|--|
| Free (`plan_name=FREE`) | `ministral-14b-2512` | `mistral-large-4` |
| Paid / EDU / other non-free | `mistral-small-2603` | `mistral-large-2512` |

You can still override models in the config or with `--model` / `--scenario-model`. If an
override is not allowed for your plan, extract fails early and suggests the plan defaults
above.

## Installation

You need Python 3.11 or newer and a [Mistral API key](https://console.mistral.ai). The same
steps work on macOS, Linux and Windows.

```bash
pip install "git+https://github.com/dvalexieva97/auto-lca-public.git"
```

Or, with [uv](https://docs.astral.sh/uv/), as a standalone command:

```bash
uv tool install "git+https://github.com/dvalexieva97/auto-lca-public.git"
```

The first install takes a few minutes: it pulls in PyTorch (about 1 GB), which is used for
section detection and similarity scoring.

Optional extras (add them in brackets, e.g. `pip install "auto-lca[eval] @ git+https://github.com/dvalexieva97/auto-lca-public.git"`):

| Extra | Adds |
|--|--|
| `eval` | BERTScore sheet in `auto-lca compare` |
| `tables` | Docling / Tabula table extraction (`auto_lca.process.nlp.tables`; Tabula needs Java) |
| `all` | All of the above |

### API keys

Set them as environment variables, or in a `.env` file in the folder you run `auto-lca` from:

```bash
MISTRAL_API_KEY="your-mistral-key"     # required: extraction and product system matching
OPENAI_API_KEY="your-openai-key"       # optional: only for `auto-lca compare --llm-judge`
```

### Try it in a clean environment

To test the package without touching your other Python setups, install it into a new virtual
environment in an empty folder:

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

auto-lca sample-config --mini --format csv    # writes sample_config_mini.csv here
auto-lca extract --pdfs baldini2020.pdf --config sample_config_mini.csv --output results
```

`results/baldini2020.json` then holds the product systems (dairy farm scenarios) and the
extracted fields. If you skip `sample-config`, `extract` copies the bundled
`sample_config_mini.csv` (or any other `sample_config*` name) into the folder itself. Delete
the folder to remove everything again.

## Quick start

```bash
auto-lca sample-config --format xlsx   # writes sample_config.xlsx here; edit the fields
auto-lca extract --pdfs papers/*.pdf --config sample_config.xlsx --output results
```

## Configuration

The configuration defines what to extract. It can be a **CSV**, **XLSX** or **JSON** file; CSV
and XLSX files are converted to JSON next to the original file.

The schema has two levels: a **field set** is a named group of **fields** that are extracted
together (e.g. "Lifecycle Assessment Characteristics"), and each field has a data type (e.g.
`str`, `float | None`). Each field set is extracted either once per paper or once per product
system (a product system is a case, pathway or scenario that the paper models).

Copy a bundled example into the current folder with `auto-lca sample-config`
(`--format json|csv|xlsx`, `--mini` for a two-field-set config).

### CSV and XLSX format

One row per field, with these columns:

- `field_set`: the field set the field belongs to
- `description`: what the field set covers (repeat it on every row of the set)
- `by_production_system`: `TRUE` to extract the field set once per product system, `FALSE` for
  once per paper
- `field`: the field name
- `data_type`: the data type (e.g. `str`, `float`, `str | None`)

An XLSX config uses the same columns in a sheet named `Field sets`, plus an optional
`Settings` sheet (`section`, `key`, `value`) for the defaults below.

**Example** (`my_config.csv`):

```csv
field_set,description,by_production_system,field,data_type
Lifecycle Assessment Characteristics,"Core LCA framework characteristics",FALSE,Functional Unit,str
Lifecycle Assessment Characteristics,"Core LCA framework characteristics",FALSE,System Boundary,str
Results,"Environmental impact assessment results",TRUE,Carbon footprint (kg CO2e),str | None
```

### JSON format

```json
{
  "defaults": {
    "extract_production_systems": true,
    "extract_non_primary_production_systems": false,
    "by_production_system": true,
    "strip_bibliography": true
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

| Setting | Default | Meaning |
|--|--|--|
| `model` | *(from Mistral plan)* | Optional override for field extraction. If omitted, AUTO-LCA uses the plan defaults in [Release notes](#october-2026--mistral-free-tier-model-defaults) |
| `production_system_model` | *(from Mistral plan)* | Optional override for identifying and classifying product systems |
| `extract_production_systems` | `true` | Identify the product systems of each paper. `false` gives one record per paper |
| `extract_non_primary_production_systems` | `false` | Also extract fields for product systems that are not the paper's own (e.g. literature comparators). When `false`, these are kept and labelled, but without fields |
| `strip_bibliography` | `true` | Remove the reference list before extraction |

Older configs that use `concepts` / `structure`, or `scenario` instead of `production_system`
in the keys, still load.

## Extract data from papers

```bash
auto-lca extract --pdfs papers/*.pdf --config my_config.csv --output results
```

| Option | Meaning |
|--|--|
| `--pdfs` | One or more PDF files (use a shell glob such as `papers/*.pdf` for a folder) |
| `--config` | Configuration (`.json`, `.csv` or `.xlsx`) |
| `--output` | Folder for results |
| `--model` / `--scenario-model` | Override plan / config models (checked against your Mistral plan) |
| `--model-profile` | `auto` (default), `paid`, or `free`: force which plan default pair to use |
| `--scenarios-only` | Identify product systems only, without field extraction |
| `--overwrite` | Re-extract papers that already have a result |

Papers that already have a result in `--output` are skipped, so an interrupted run can be
resumed by running the same command again.

**Output:**

- `results/<paper id>.json`: one file per PDF (the paper ID is the PDF file name), with the
  product systems, the extracted fields, token usage and run metadata.
- `results/extracted_sections/`: the detected paper sections (Methods, Results, …) per PDF.

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
| `sample-config` | Copy a bundled example configuration into the current folder |
| `reclassify-scenarios` | Re-label product system categories on existing extracts |
| `extract-sections` | Rebuild `extracted_sections/` for existing extracts |
| `enrich-excel` | Verify missing reference fields against the PDFs and fill them in |
| `regenerate-stats` | Recompute statistics from an existing joined workbook |

Run `auto-lca <command> --help` for the options. `python -m auto_lca …` works the same as
`auto-lca …`.

## Developing and running the tests

```bash
git clone https://github.com/dvalexieva97/auto-lca-public.git
cd auto-lca-public
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The tests need no API keys and make no API calls.

## Troubleshooting

**"MISTRAL_API_KEY environment variable not set" or authentication errors:** set
`MISTRAL_API_KEY` in your environment, or in a `.env` file in the folder you run `auto-lca` from.

**Models not usable / tier errors on extract:** your key's Mistral plan cannot call the
models in the config (or the CLI overrides). Remove those overrides so plan defaults apply,
or switch to a paid plan. See [Release notes](#october-2026--mistral-free-tier-model-defaults).

**"Config file not found":** check the path to your config and that it ends in `.json`, `.csv`
or `.xlsx`.

**A PDF fails:** check that it is not corrupted and contains text (not only scanned images). A
paper that fails is logged and skipped; rerun the same command to retry only the missing ones.
