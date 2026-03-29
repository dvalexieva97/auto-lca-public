import argparse
import os

import pandas as pd

from auto_lca.process.sog.mistral_sog import MistralStructuredExtractor
from auto_lca.statistics.compare import join_all_json_results


def main():
    """
    Main entry point for SOG runner.

    Accepts:
    - List of PDF paths (or folder containing PDFs)
    - Config path (CSV or JSON)
    - Output folder
    """
    parser = argparse.ArgumentParser(
        description="Run Structured Output Generation (SOG) on PDF files"
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        help="List of PDF file paths, or a single folder path containing PDFs",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/auto_lca/process/sog/sample_config.csv",
        help="Path to config file (CSV or JSON format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/auto_lca/data/output/results/",
        help="Output folder for results",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process PDFs even if output already exists",
    )

    args = parser.parse_args()

    # Determine PDF paths
    pdf_paths = []
    for path in args.pdfs:
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

    if not pdf_paths:
        print("Error: No PDF files found to process")
        return

    # Initialize extractor with config (handles CSV/JSON conversion automatically)
    print(f"Loading config from: {args.config}")
    extractor = MistralStructuredExtractor(config_path=args.config)

    # Process PDFs in batch
    results = extractor.process_pdfs_batch(
        pdf_paths=pdf_paths,
        output_folder=args.output,
        skip_existing=not args.no_skip_existing,
    )

    # Save failure report if there are any failures
    if results["fails"]:
        failed_df = pd.DataFrame(results["fails"])
        fails_csv_path = os.path.join(args.output, "fails.csv")
        failed_df.to_csv(fails_csv_path, index=False)
        print(
            f"\n{len(results['fails'])} PDFs failed. Failure report saved to: {fails_csv_path}"
        )

    print("\nProcessing complete!")
    print(f"Successfully processed: {len(results['success'])} PDFs")
    print(f"Failed: {len(results['fails'])} PDFs")
    print(f"Results saved to: {args.output}")

    # Always join all JSON results into a CSV
    join_all_json_results(args.output)


if __name__ == "__main__":
    main()


# /Users/dvalexieva/Documents/scripts/uni/auto-lca/auto-lca-env/bin/python /Users/dvalexieva/Documents/scripts/uni/auto-lca/src/auto_lca/scripts/sog_runner.py --pdfs "src/auto_lca/data/input-papers/sample"
