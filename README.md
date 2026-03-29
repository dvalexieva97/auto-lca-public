# AUTO-LCA

**AUTO-LCA** is an automated Life Cycle Assessment (LCA) database generation tool that crawls academic LCA research papers and extracts structured data from them using machine learning. The tool leverages Mistral AI's structured output generation to extract relevant information from PDF documents.

## What It Does

AUTO-LCA automates the process of:
- **Searching** for academic papers on LCA topics using Semantic Scholar. This feature is currently under development
- **Structured Output Generation (SOG)** Extracting structured data from research paper PDFs
- **Aggregating** datasets from multiple papers into a unified format

The tool uses Mistral AI's structured output generation to extract specific concepts and fields from papers based on a configurable schema, making it easy to build databases from academic literature.

## Setup

### Prerequisites
In order to set up this repo, you need the following:
- git
- make
- pyenv
- python 3.13.7


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
  - **Recommended**: Use the automated setup script (`setup-windows.ps1`) which installs Git, make and pyenv automatically
  - **Manual**: Download and install from [https://git-scm.com/download/win](https://git-scm.com/download/win)
  - **Or**: Use Chocolatey: `choco install git`
  
  **Important for Windows**: When installing Git for Windows, make sure to:
  - Select "Git from the command line and also from 3rd-party software" during installation
  - This ensures `make` is available via Git's MSYS installation

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
   git clone https://github.com/dvalexieva97/auto-lca-public.git
   cd auto-lca-public
   ```

2. **Initialize the project**:

   Choose one of the following methods based on your operating system:

   #### Option A: Automated Setup (Recommended)

   **Windows (PowerShell)**:
   ```powershell
   # Run the automated setup script (handles all prerequisites)
   .\setup-windows.ps1
   ```
   
   This script will automatically:
   - Install Chocolatey (if needed)
   - Install Git, Make, and pyenv-win
   - Install Python 3.13.7
   - Create a virtual environment
   - Install all dependencies
   
   **Note**: You may need to run PowerShell as Administrator for some installations.
   
   **macOS/Linux**:
   ```bash
   make init
   ```
      
   This Python script works on all platforms and provides an interactive setup process.

   #### Option B: Manual Setup

   If you prefer to set up manually or the automated scripts don't work:

   **Windows**:
   ```powershell
   # 1. Install prerequisites manually (see Windows Setup Guide below)
   # 2. Then run:
   make init
   ```
   
   **macOS/Linux**:
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
python -m auto_lca.scripts.sog_runner --pdfs src/auto_lca/data/input-papers/  --config src/auto_lca/process/sog/sample_config.csv
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

### Setting Up VSCode / Cursor IDE

To ensure VSCode or Cursor uses the correct virtual environment for debugging and terminal execution:

#### Automatic Setup (Optional)

If you're using VSCode or Cursor IDE, you can configure it to use the virtual environment:

```bash
make set-vscode-python
```

This will automatically:
- Create `.vscode/settings.json` (works for both VSCode and Cursor)
- Set the Python interpreter to your virtual environment
- Configure the IDE to use the correct environment

**Note**: This step is optional. If you're not using VSCode/Cursor (e.g., running in a VM, using a different editor, or running from command line), you can skip this step.

#### Manual Setup

If you prefer to set it up manually:

1. **Open Command Palette**:
   - VSCode/Cursor: `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)

2. **Select Python Interpreter**:
   - Type: `Python: Select Interpreter`
   - Choose: `./auto-lca-env/bin/python` (macOS/Linux) or `.\auto-lca-env\Scripts\python.exe` (Windows)

3. **Verify Terminal Uses Venv**:
   - Open a new terminal in the IDE
   - The terminal should automatically activate the virtual environment
   - You should see `(auto-lca-env)` in your terminal prompt

4. **For Debugging**:
   - Create or edit `.vscode/launch.json`:
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: Current File",
         "type": "debugpy",
         "request": "launch",
         "program": "${file}",
         "console": "integratedTerminal",
         "justMyCode": true,
         "python": "${workspaceFolder}/auto-lca-env/bin/python"
       }
     ]
   }
   ```
   - **Windows users**: Use `"python": "${workspaceFolder}/auto-lca-env/Scripts/python.exe"` instead

5. **Verify Setup**:
   - Open any Python file
   - Check the bottom-right corner of the IDE - it should show the Python version from your venv
   - Run a Python file with `F5` - it should use the virtual environment

#### Troubleshooting IDE Setup

**Python interpreter not found**:
- Make sure you've run `make init` first
- Verify the venv exists: `ls auto-lca-env/bin/python` (macOS/Linux) or `dir auto-lca-env\Scripts\python.exe` (Windows)
- Re-run `make set-vscode-python`

**Terminal doesn't activate venv**:
- Close and reopen the terminal
- Manually activate: `source auto-lca-env/bin/activate` (macOS/Linux) or `.\auto-lca-env\Scripts\Activate.ps1` (Windows)

**Debugging uses wrong Python**:
- Check `.vscode/launch.json` has the correct path
- Use the Command Palette to select the interpreter again
- Restart the IDE

### Output

After processing, you'll find:

- **JSON files**: One per PDF with extracted structured data
- **CSV file**: A joined CSV file with all results combined
- **fails.csv** (if any failures): A report of PDFs that failed to process

Results are saved in the specified output folder (default: `src/auto_lca/data/output/results/`).

## Windows Setup Guide

> **📘 For detailed Windows setup instructions, see [WINDOWS_SETUP.md](WINDOWS_SETUP.md)**

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

### Windows-specific issues

- See the [Windows Setup Guide](#windows-setup-guide) above for detailed troubleshooting
- If `make init` fails, try using `python setup.py` instead
- Ensure all prerequisites are installed before running setup
