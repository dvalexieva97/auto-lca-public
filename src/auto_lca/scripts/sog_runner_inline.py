import os

import pandas as pd

from auto_lca.process.sog.mistral_sog import MistralStructuredExtractor
from auto_lca.statistics.compare import join_all_json_results

# ============================================================================
# CONFIGURATION VARIABLES - Edit these to configure the runner
# ============================================================================

# PDF paths: List of PDF file paths, or folder paths containing PDFs
# Examples:
#   PDF_PATHS = ["path/to/file1.pdf", "path/to/file2.pdf"]
#   PDF_PATHS = ["src/auto_lca/data/input-papers/sample"]
#   PDF_PATHS = ["path/to/folder1", "path/to/file.pdf"]
PDF_PATHS = ["src/auto_lca/data/input-papers/sample"]

# Config file path (CSV or JSON format)
CONFIG_PATH = "src/auto_lca/process/sog/sample_config.csv"

# Output folder for results
OUTPUT_FOLDER = "src/auto_lca/data/output/results/"

# If True, skip PDFs that already have output files
# If False, reprocess all PDFs even if output exists
SKIP_EXISTING = True

# ============================================================================


def determine_pdf_paths(paths):
    """
    Determine all PDF paths from a list of paths (files or directories).

    Args:
        paths: List of file paths or directory paths

    Returns:
        List of PDF file paths
    """
    pdf_paths = []
    for path in paths:
        if os.path.isdir(path):
            # If it's a directory, find all PDFs in it
            pdf_paths.extend(
                [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(".pdf")
                ]
            )
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            pdf_paths.append(path)
        else:
            print(f"Warning: {path} is not a valid PDF file or directory, skipping")

    return pdf_paths


def main():
    """
    Main entry point for SOG runner with in-code configuration.
    """
    # Determine PDF paths
    pdf_paths = determine_pdf_paths(PDF_PATHS)

    if not pdf_paths:
        print("Error: No PDF files found to process")
        return

    # Initialize extractor with config (handles CSV/JSON conversion automatically)
    print(f"Loading config from: {CONFIG_PATH}")
    extractor = MistralStructuredExtractor(config_path=CONFIG_PATH)

    # Process PDFs in batch
    results = extractor.process_pdfs_batch(
        pdf_paths=pdf_paths,
        output_folder=OUTPUT_FOLDER,
        skip_existing=SKIP_EXISTING,
    )

    # Save failure report if there are any failures
    if results["fails"]:
        failed_df = pd.DataFrame(results["fails"])
        fails_csv_path = os.path.join(OUTPUT_FOLDER, "fails.csv")
        failed_df.to_csv(fails_csv_path, index=False)
        print(
            f"\n{len(results['fails'])} PDFs failed. Failure report saved to: {fails_csv_path}"
        )

    print("\nProcessing complete!")
    print(f"Successfully processed: {len(results['success'])} PDFs")
    print(f"Failed: {len(results['fails'])} PDFs")
    print(f"Results saved to: {OUTPUT_FOLDER}")

    # Always join all JSON results into a CSV
    join_all_json_results(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
