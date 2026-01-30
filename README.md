# AUTO-LCA

**AUTO-LCA** is an automated Life Cycle Assessment (LCA) database generation tool that crawls academic LCA research papers and extracts structured data from them using machine learning. The tool uses [Semantic Scholar](https://www.semanticscholar.org/) to search for papers and leverages Mistral AI's structured output generation to extract relevant information from PDF documents.

## What It Does

AUTO-LCA automates the process of:

- **Searching** for academic papers on LCA topics using Semantic Scholar. This feature is currently under development
- **Structured Output Generation (SOG)** Extracting structured data from research paper PDFs
- **Aggregating** datasets from multiple papers into a unified format

The tool uses Mistral AI's structured output generation to extract specific concepts and fields from papers based on a configurable schema, making it easy to build databases from academic literature.

## Setup

### Prerequisites

#### Installing Git

If you don't have Git installed, you'll need to install it first:

- **macOS**: 
  ```bash
  # Using Homebrew (recommended)
  brew install git
  
  # Or download from https://git-scm.com/download/mac
  ```

- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update
  sudo apt install git
  ```

- **Windows**: 
  - Download and install from [https://git-scm.com/download/win](https://git-scm.com/download/win)
  - Or use a package manager like Chocolatey: `choco install git`

After installation, verify Git is installed:
```bash
git --version
```

**Note**: If you're cloning the repository for the first time, you may also want to configure Git with your name and email:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd auto-lca-public
   ```

2. **Initialize the project**:
   ```bash
   make init
   ```

   This will set up the virtual environment and install all dependencies automatically.

### Getting a Mistral API Token

1. **Sign up** for a Mistral AI account at [https://mistral.ai](https://mistral.ai)
2. **Navigate** to the API section in your dashboard
3. **Create** a new API key
4. **Add your API key to the `.env` file**:
   ```bash
   echo 'MISTRAL_API_KEY="your-api-key-here"' >> .env
   ```

   **Note**: The tool requires the `MISTRAL_API_KEY` environment variable to be set. Without it, the extraction will fail.

## Configuration

AUTO-LCA uses a configuration file to define what data to extract from papers. You can use either **CSV** or **JSON** format.

### CSV Configuration Format

The CSV config will automatically be convert to .json. The CSV config file should have the following columns:

- `name`: The concept/field name to extract
- `description`: Detailed description of what to extract
- `aspect`: The aspect category (e.g., "Methods_Results")
- `by_scenario`: Whether the field varies by scenario (TRUE/FALSE)
- `structure_title`: The field name within the structure
- `structure_data_type`: The data type (e.g., "str", "float", "str | None")

**Example CSV config** (`my_config.csv`):
```csv
name,description,aspect,by_scenario,structure_title,structure_data_type
Lifecycle Assessment Characteristics,"Core LCA framework characteristics",Methods_Results,FALSE,Functional Unit,str
Lifecycle Assessment Characteristics,"Core LCA framework characteristics",Methods_Results,FALSE,System Boundary,str
Results,"Environmental impact assessment results",Methods_Results,TRUE,Carbon footprint (kg CO2e),str | None
```

### JSON Configuration Format

The JSON config follows this structure:
```json
{
  "defaults": {
    "add_references": false,
    "add_justifications": false,
    "llm_role": "reasoner_text",
    "singular_occurrence": false,
    "by_scenario": true,
    "aspect": "Methods_Results"
  },
  "aspects": [
    {
      "name": "Methods_Results",
      "description": "Methods_Results"
    }
  ],
  "concepts": [
    {
      "name": "Lifecycle Assessment Characteristics",
      "description": "Core LCA framework characteristics...",
      "aspect": "Methods_Results",
      "by_scenario": false,
      "structure": {
        "Functional Unit": "str",
        "System Boundary": "str"
      }
    }
  ]
}
```

**Note**: If you provide a CSV config, it will automatically be converted to JSON format internally. You can find example configs at:

- `src/auto_lca/process/sog/sample_config.csv`
- `src/auto_lca/process/sog/sample_config.json`

## Running AUTO-LCA

### Basic Usage

Process PDF files using the main script:

```bash
python -m auto_lca.scripts.sog_runner \
  --pdfs path/to/your/pdfs \
  --config path/to/your/config.csv \
  --output path/to/output/folder
```

### Command-Line Options

- `--pdfs`: PDF file(s) or folder containing PDFs (required)
  - Can specify multiple files: `--pdfs file1.pdf file2.pdf`
  - Or a folder: `--pdfs path/to/folder/`
- `--config`: Path to your config file (CSV or JSON) (default: `src/auto_lca/process/sog/sample_config.csv`)
- `--output`: Output folder for results (default: `src/auto_lca/data/output/results/`)
- `--no-skip-existing`: Process PDFs even if output already exists

### Examples

**Process a single PDF**:
```bash
python -m auto_lca.scripts.sog_runner \
  --pdfs paper.pdf \
  --config my_config.csv \
  --output results/
```

**Process all PDFs in a folder**:
```bash
python -m auto_lca.scripts.sog_runner \
  --pdfs src/auto_lca/data/input-papers/ \
  --config src/auto_lca/process/sog/sample_config.csv
```

**Process multiple specific PDFs**:
```bash
python -m auto_lca.scripts.sog_runner \
  --pdfs paper1.pdf paper2.pdf paper3.pdf \
  --config my_config.json \
  --output custom_output/
```

### Using the Inline Runner (No Command-Line Arguments)

If you prefer not to use command-line arguments, you can use the inline runner script instead. Simply edit the configuration variables directly in the file:

1. **Open** `src/auto_lca/scripts/sog_runner_inline.py`

2. **Edit the configuration variables** at the top of the file:
   ```python
   # PDF paths: List of PDF file paths, or folder paths containing PDFs
   PDF_PATHS = ["src/auto_lca/data/input-papers/sample"]
   
   # Config file path (CSV or JSON format)
   CONFIG_PATH = "src/auto_lca/process/sog/sample_config.csv"
   
   # Output folder for results
   OUTPUT_FOLDER = "src/auto_lca/data/output/results/"
   
   # If True, skip PDFs that already have output files
   SKIP_EXISTING = True
   ```

3. **Run the script**:
   ```bash
   python src/auto_lca/scripts/sog_runner_inline.py
   ```

This approach is useful if you want to:
- Avoid typing long command-line arguments
- Save your configuration directly in the script
- Run the script from an IDE or notebook environment

### Output

After processing, you'll find:

- **JSON files**: One per PDF with extracted structured data
- **CSV file**: A joined CSV file with all results combined
- **fails.csv** (if any failures): A report of PDFs that failed to process

Results are saved in the specified output folder (default: `src/auto_lca/data/output/results/`).

## Troubleshooting

### "MISTRAL_API_KEY environment variable not set"

- Make sure you've added your API key to the `.env` file: `MISTRAL_API_KEY="your-key"`
- Verify the `.env` file exists and contains the key

### "Config file not found"

- Check that the path to your config file is correct
- Ensure the file has the correct format (CSV or JSON)

### PDF processing fails

- Check that PDFs are not corrupted
- Ensure PDFs contain extractable text (not just images)
- Review the `fails.csv` file in the output folder for error details
