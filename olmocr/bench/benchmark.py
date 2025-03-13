#!/usr/bin/env python3
"""
This script runs olmocr bench.
It will take as an argument a folder, and scan it for .jsonl files which contain the various rules and properties that we will check.
It will then validate the JSON files to make sure they are all valid.
Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate.
We will validate that each one of those contains at least one .md file (or repeated generations, e.g. _pg{page}_repeat{repeat}.md)
corresponding to its parse for every .pdf in the /pdfs folder.
Then, we will read each one, and check if they pass against all the rules.
If a rule fails on some of the repeats, a short explanation is printed.
The final score is averaged over the repeated generations.
Statistical analysis including bootstrap confidence intervals are provided for the results.
Pairwise permutation tests are conducted between specific candidate pairs.
"""

import argparse
import glob
import os
import sys
import re

from typing import Dict, List, Tuple, Optional
from pypdf import PdfReader

from .tests import BasePDFTest, BaselineTest, load_tests
from .katex.render import clear_cache_dir
from .utils import calculate_bootstrap_ci, perform_permutation_test

def evaluate_candidate(
    candidate_folder: str, all_tests: List[BasePDFTest], pdf_basenames: List[str], force: bool=False
) -> Tuple[float, int, List[str], List[str], Dict[str, List[float]], List[float]]:
    """
    For the candidate folder (pipeline tool output), validate that it contains at least one .md file
    (i.e. repeated generations like _pg{page}_repeat{repeat}.md) for every PDF in the pdf folder.
    Then, run each rule against all corresponding .md files and average the results.

    Returns a tuple:
      (overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown, all_test_scores)

      - overall_score: Average fraction of tests passed (averaged over repeats and tests).
      - total_tests: Total number of tests evaluated.
      - candidate_errors: List of candidate errors (e.g. missing files).
      - test_failures: List of failure messages for tests not passing on all repeats.
      - test_type_breakdown: Dictionary mapping test type to list of average pass ratios for tests of that type.
      - all_test_scores: List of all individual test scores (used for bootstrapping).
    """
    candidate_errors = []
    test_failures = []
    test_type_breakdown = {}  # key: test type, value: list of average pass ratios
    all_test_scores = []  # Store all individual test scores for bootstrapping
    candidate_name = os.path.basename(candidate_folder)

    # Map each PDF to its corresponding MD repeats (e.g., doc1_pg1_repeat1.md, doc1_pg2_repeat2.md, etc.)
    pdf_to_md_files = {}
    for pdf_name in pdf_basenames:
        md_base = os.path.splitext(pdf_name)[0]
        # Updated regex for new format: {pdf_name}_pg<page>_repeat<repeat>.md
        md_regex = re.compile(rf"^{re.escape(md_base)}_pg\d+_repeat\d+\.md$")
        
        # List all files in the candidate folder and filter using regex
        all_files = os.listdir(candidate_folder)
        md_files = [os.path.join(candidate_folder, f) for f in all_files if md_regex.match(f)]
        
        if not md_files and not force:
            candidate_errors.append(
                f"Candidate '{candidate_name}' is missing MD repeats for {pdf_name} "
                f"(expected files matching {md_base}_pg{{page}}_repeat*.md)."
            )
        else:
            pdf_to_md_files[pdf_name] = md_files

    if candidate_errors:
        return (0.0, len(all_tests), candidate_errors, test_failures, test_type_breakdown, all_test_scores)

    total_test_score = 0.0

    # Evaluate each test. Each test references a PDF (e.g., "doc1.pdf") and a specific page.
    for test in all_tests:
        test_type = test.type
        if test_type not in test_type_breakdown:
            test_type_breakdown[test_type] = []
        pdf_name = test.pdf
        md_base = os.path.splitext(pdf_name)[0]
        md_files = pdf_to_md_files.get(pdf_name, [])
        # Filter MD files for the specific page corresponding to the test
        page_md_files = [f for f in md_files if re.search(rf"_pg{test.page}_", os.path.basename(f))]
        if not page_md_files:
            candidate_errors.append(
                f"Candidate '{candidate_name}' is missing MD repeats for {pdf_name} page {test.page} "
                f"(expected files matching {md_base}_pg{test.page}_repeat*.md)."
            )
            continue

        repeat_passes = 0
        num_repeats = 0
        explanations = []
        for md_path in page_md_files:
            num_repeats += 1
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                candidate_errors.append(f"Error reading {md_path}: {e}")
                continue

            try:
                # Use the test's run method to evaluate the content
                passed, explanation = test.run(md_content)
                if passed:
                    repeat_passes += 1
                else:
                    explanations.append(explanation)
            except Exception as e:
                candidate_errors.append(f"Error running test {test.id} on {md_path}: {e}")
                explanations.append(str(e))

        test_avg = repeat_passes / num_repeats if num_repeats > 0 else 0.0
        all_test_scores.append(test_avg)  # Add to list for bootstrapping
        total_test_score += test_avg
        if test_avg < 1.0:
            test_failures.append(
                f"Test {test.id} on {md_base} page {test.page} average pass ratio: {test_avg:.3f} "
                f"({repeat_passes}/{num_repeats} repeats passed). Ex: {explanations[0] if explanations else 'No explanation'}"
            )
        test_type_breakdown[test_type].append(test_avg)

    overall_score = total_test_score / len(all_tests) if all_tests else 0.0
    return (overall_score, len(all_tests), candidate_errors, test_failures, test_type_breakdown, all_test_scores)


def main():
    parser = argparse.ArgumentParser(description="Run OLMOCR Bench.")
    parser.add_argument(
        "--input_folder",
        default=os.path.join(os.path.dirname(__file__), "sample_data"),
        help="Path to the folder containing .jsonl files, /pdfs folder, and pipeline tool subfolders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run benchmark even if some files are missing",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Run test only for a single candidate"
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence interval calculation (default: 1000).",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
        help="Confidence level for interval calculation (default: 0.95 for 95% CI).",
    )
    parser.add_argument(
        "--permutation_tests",
        action="store_true",
        help="Run permutation testing",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    n_bootstrap = args.bootstrap_samples
    ci_level = args.confidence_level
    pdf_folder = os.path.join(input_folder, "pdfs")

    # Check that the pdfs folder exists
    if not os.path.exists(pdf_folder):
        print("Error: /pdfs folder must exist in your data directory.", file=sys.stderr)
        sys.exit(1)

    # Find all pdf files in the pdf folder
    all_pdf_files = list(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    if not all_pdf_files:
        print(f"Error: No PDF files found in {pdf_folder}", file=sys.stderr)
        sys.exit(1)

    # Get PDF basenames (e.g. "doc1.pdf")
    pdf_basenames = [os.path.basename(p) for p in all_pdf_files]

    # Find and validate .jsonl files in the input folder
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {input_folder}.", file=sys.stderr)
        sys.exit(1)

    # Load and concatenate all test rules from JSONL files
    all_tests = []
    for jsonl_path in jsonl_files:
        tests = load_tests(jsonl_path)
        all_tests.extend(tests)

    if not all_tests:
        print("No valid tests found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Add in a default repeat test for every PDF that doesn't already have one
    for pdf in pdf_basenames:
        if not any(t.type == "baseline" for t in all_tests if t.pdf == pdf):
            all_tests.append(BaselineTest(id=f"{pdf}_baseline", pdf=pdf, page=1, type="baseline"))

    # Make sure that each PDF and page has at least one test in it
    for pdf in pdf_basenames:
        pdf_doc = PdfReader(os.path.join(pdf_folder, pdf))
        
        for page in range(1, len(pdf_doc.pages) + 1):
            if not any(test for test in all_tests if test.pdf == pdf and test.page == page) and not args.force:
                print(f"No dataset entry found for pdf {pdf} page {page}")
                sys.exit(1)

    # Identify candidate pipeline folders (subdirectories of input_folder excluding /pdfs)
    candidate_folders = []
    for entry in os.listdir(input_folder):
        full_path = os.path.join(input_folder, entry)
        if args.candidate is not None:
            if entry == args.candidate:
                candidate_folders.append(full_path) 
        else:
            if os.path.isdir(full_path) and entry != "pdfs":
                candidate_folders.append(full_path)

    if not candidate_folders:
        print("Error: No candidate pipeline folders found (subdirectories besides 'pdfs').", file=sys.stderr)
        sys.exit(1)

    candidate_folders.sort()

    # Evaluate each candidate
    summary = []
    print("\nRunning tests for each candidate:")
    for candidate in candidate_folders:
        candidate_name = os.path.basename(candidate)
        overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown, all_test_scores = evaluate_candidate(
            candidate, all_tests, pdf_basenames, args.force,
        )
        
        # Calculate confidence interval
        if all_test_scores:
            ci = calculate_bootstrap_ci(all_test_scores, n_bootstrap=n_bootstrap, ci_level=ci_level)
        else:
            ci = (0.0, 0.0)
            
        summary.append((candidate_name, overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown, ci, all_test_scores))
        
        print(f"\nCandidate: {candidate_name}")
        if candidate_errors:
            for err in candidate_errors:
                print(f"  [ERROR] {err}")
        else:
            if test_failures:
                for fail in test_failures:
                    print(f"  [FAIL] {fail}")
            print(f"  Average Score: {overall_score * 100:.1f}% (95% CI: [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]) over {total_tests} tests.")

    # Print final summary with breakdown by test type
    print("\n" + "=" * 60)
    print("Final Summary with 95% Confidence Intervals:")
    for candidate_name, overall_score, total_tests, candidate_errors, _, test_type_breakdown, ci, _ in summary:
        if candidate_errors:
            status = "FAILED (errors)"
            ci_str = "N/A"
            ciw_str = ""
        else:
            status = f"{overall_score * 100:0.1f}%"
            half_width = ((ci[1] - ci[0]) / 2) * 100
            ciw_str = f"± {half_width:0.1f}%"
            ci_str = f"[{ci[0] * 100:0.1f}%, {ci[1] * 100:0.1f}%]"
        
        print(f"{candidate_name:20s} : Average Score: {status} {ciw_str}")
        
        for ttype, scores in test_type_breakdown.items():
            if scores:
                avg = sum(scores) / len(scores) * 100
            else:
                avg = 0.0
            
            print(f"    {ttype:8s}: {avg:0.1f}% average pass rate over {len(scores)} tests")
        
        print("")
    
    # Perform pairwise permutation tests
    if args.permutation_tests:
        print("\n" + "=" * 60)
        print("Pairwise Permutation Tests:")
        
        valid_candidates = [c for c in summary if not c[3]]  # Filter out candidates with errors
        olmocr_candidates = sorted([c for c in valid_candidates if "olmocr" in c[0].lower()], key=lambda x: x[1], reverse=True)
        non_olmocr_candidates = sorted([c for c in valid_candidates if "olmocr" not in c[0].lower()], key=lambda x: x[1], reverse=True)
        
        top_olmocr = olmocr_candidates[0] if olmocr_candidates else None
        top_non_olmocr = non_olmocr_candidates[0] if non_olmocr_candidates else None
        top_two_olmocr = olmocr_candidates[:2]

        # Test 1: Top olmocr vs Top non-olmocr
        if top_olmocr and top_non_olmocr:
            olmocr_name, olmocr_score = top_olmocr[0], top_olmocr[1]
            non_olmocr_name, non_olmocr_score = top_non_olmocr[0], top_non_olmocr[1]
            olmocr_scores = top_olmocr[7]  # all_test_scores
            non_olmocr_scores = top_non_olmocr[7]  # all_test_scores
            
            diff, p_value = perform_permutation_test(
                olmocr_scores, non_olmocr_scores
            )
            
            print(f"\nComparison 1: Top olmocr vs Top non-olmocr candidate")
            print(f"  {olmocr_name} ({olmocr_score*100:.1f}%) vs {non_olmocr_name} ({non_olmocr_score*100:.1f}%)")
            print(f"  Difference: {diff*100:.2f}% (positive means {olmocr_name} is better)")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  Result: Statistically significant difference (p < 0.05)")
            else:
                print(f"  Result: No statistically significant difference (p ≥ 0.05)")
        else:
            print("\nCannot perform olmocr vs non-olmocr comparison: Missing candidates")
        
        # Test 2: Top two olmocr candidates (if there are at least two)
        if len(top_two_olmocr) >= 2:
            olmocr1_name, olmocr1_score = top_two_olmocr[0][0], top_two_olmocr[0][1]
            olmocr2_name, olmocr2_score = top_two_olmocr[1][0], top_two_olmocr[1][1]
            olmocr1_scores = top_two_olmocr[0][7]  # all_test_scores
            olmocr2_scores = top_two_olmocr[1][7]  # all_test_scores
            
            diff, p_value = perform_permutation_test(
                olmocr1_scores, olmocr2_scores
            )
            
            print(f"\nComparison 2: Top two olmocr candidates")
            print(f"  {olmocr1_name} ({olmocr1_score*100:.1f}%) vs {olmocr2_name} ({olmocr2_score*100:.1f}%)")
            print(f"  Difference: {diff*100:.2f}% (positive means {olmocr1_name} is better)")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  Result: Statistically significant difference (p < 0.05)")
            else:
                print(f"  Result: No statistically significant difference (p ≥ 0.05)")
        else:
            print("\nCannot perform top two olmocr comparison: Not enough olmocr candidates")
        
        print("=" * 60)


if __name__ == "__main__":
    main()
