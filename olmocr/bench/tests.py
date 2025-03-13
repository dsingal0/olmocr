import json
import re
import numpy as np
from bs4 import BeautifulSoup

from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz
from tqdm import tqdm

from olmocr.repeatdetect import RepeatDetector
from .katex.render import render_equation, compare_rendered_equations

class TestType(str, Enum):
    BASELINE = "baseline"
    PRESENT = "present"
    ABSENT = "absent"
    ORDER = "order"
    TABLE = "table"
    MATH = "math"


class TestChecked(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


class ValidationError(Exception):
    """Exception raised for validation errors."""

    pass


def normalize_text(md_content: str) -> str:
    # Normalize whitespace in the md_content
    md_content = re.sub(r'\s+', " ", md_content)

    # Dictionary of characters to replace: keys are fancy characters, values are ASCII equivalents
    replacements = {
        "‘": "'", "’": "'", "‚": "'",
        "“": "\"", "”": "\"", "„": "\"",
        "＿": "_",
        "–": "-", "—": "-", "‑": "-", "‒": "-"
    }

     # Apply all replacements from the dictionary
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    return md_content


@dataclass(kw_only=True)
class BasePDFTest:
    """
    Base class for all PDF test types.

    Attributes:
        pdf: The PDF filename.
        page: The page number for the test.
        id: Unique identifier for the test.
        type: The type of test.
        threshold: A float between 0 and 1 representing the threshold for fuzzy matching.
    """

    pdf: str
    page: int
    id: str
    type: str
    max_diffs: int = 0
    checked: Optional[TestChecked] = None

    def __post_init__(self):
        if not self.pdf:
            raise ValidationError("PDF filename cannot be empty")
        if not self.id:
            raise ValidationError("Test ID cannot be empty")
        if not isinstance(self.max_diffs, int) or self.max_diffs < 0:
            raise ValidationError(f"Max diffs must be positive number or 0")
        if self.type not in {t.value for t in TestType}:
            raise ValidationError(f"Invalid test type: {self.type}")

    def run(self, md_content: str) -> Tuple[bool, str]:
        """
        Run the test on the provided markdown content.

        Args:
            md_content: The content of the .md file.

        Returns:
            A tuple (passed, explanation) where 'passed' is True if the test passes,
            and 'explanation' provides details when the test fails.
        """
        raise NotImplementedError("Subclasses must implement the run method")


@dataclass
class TextPresenceTest(BasePDFTest):
    """
    Test to verify the presence or absence of specific text in a PDF.

    Attributes:
        text: The text string to search for.
    """

    text: str
    case_sensitive: bool=True

    def __post_init__(self):
        super().__post_init__()
        if self.type not in {TestType.PRESENT.value, TestType.ABSENT.value}:
            raise ValidationError(f"Invalid type for TextPresenceTest: {self.type}")
        if not self.text.strip():
            raise ValidationError("Text field cannot be empty")

    def run(self, md_content: str) -> Tuple[bool, str]:
        reference_query = self.text

        # Normalize whitespace in the md_content
        md_content = normalize_text(md_content)

        if not self.case_sensitive:
            reference_query = reference_query.lower()
            md_content = md_content.lower()

        # Threshold for fuzzy matching derived from max_diffs
        threshold = 1.0 - (self.max_diffs / (len(reference_query) if len(reference_query) > 0 else 1))
        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0

        if self.type == TestType.PRESENT.value:
            if best_ratio >= threshold:
                return True, ""
            else:
                msg = f"Expected '{reference_query[:40]}...' with threshold {threshold} " f"but best match ratio was {best_ratio:.3f}"
                return False, msg
        else:  # ABSENT
            if best_ratio < threshold:
                return True, ""
            else:
                msg = f"Expected absence of '{reference_query[:40]}...' with threshold {threshold} " f"but best match ratio was {best_ratio:.3f}"
                return False, msg


@dataclass
class TextOrderTest(BasePDFTest):
    """
    Test to verify that one text appears before another in a PDF.

    Attributes:
        before: The text expected to appear first.
        after: The text expected to appear after the 'before' text.
    """

    before: str
    after: str

    def __post_init__(self):
        super().__post_init__()
        if self.type != TestType.ORDER.value:
            raise ValidationError(f"Invalid type for TextOrderTest: {self.type}")
        if not self.before.strip():
            raise ValidationError("Before field cannot be empty")
        if not self.after.strip():
            raise ValidationError("After field cannot be empty")

    def run(self, md_content: str) -> Tuple[bool, str]:
        md_content = normalize_text(md_content)

        before_matches = find_near_matches(self.before, md_content, max_l_dist=self.max_diffs)
        after_matches = find_near_matches(self.after, md_content, max_l_dist=self.max_diffs)

        if not before_matches:
            return False, f"'before' text '{self.before[:40]}...' not found with max_l_dist {self.max_diffs}"
        if not after_matches:
            return False, f"'after' text '{self.after[:40]}...' not found with max_l_dist {self.max_diffs}"

        for before_match in before_matches:
            for after_match in after_matches:
                if before_match.start < after_match.start:
                    return True, ""
        return False, (f"Could not find a location where '{self.before[:40]}...' appears before " f"'{self.after[:40]}...'.")



@dataclass
class TableTest(BasePDFTest):
    """
    Test to verify certain properties of a table are held, namely that some cells appear relative to other cells correctly
    """
    # This is the target cell, which must exist in at least one place in the table
    cell: str

    # These properties say that the cell immediately up/down/left/right of the target cell has the string specified
    up: str = ""
    down: str = ""
    left: str = ""
    right: str = ""

    # These properties say that the cell all the way up, or all the way left of the target cell (ex. headings) has the string value specified
    top_heading: str = ""
    left_heading: str = ""


    def __post_init__(self):
        super().__post_init__()
        if self.type != TestType.TABLE.value:
            raise ValidationError(f"Invalid type for TableTest: {self.type}")

    def parse_markdown_tables(self, md_content: str) -> List[np.ndarray]:
        """
        Extract and parse all markdown tables from the provided content.

        Args:
            md_content: The markdown content containing tables

        Returns:
            A list of numpy arrays, each representing a parsed table
        """
        import re
        import numpy as np

        # Updated regex to allow optional leading and trailing pipes
        table_pattern = (
            r'(\|?(?:[^|\n]*\|)+[^|\n]*\|?)\s*\n'
            r'\|?(?:[ :-]+\|)+[ :-]+\|?\s*\n'
            r'((?:\|?(?:[^|\n]*\|)+[^|\n]*\|?\s*\n)+)'
        )
        table_matches = re.finditer(table_pattern, md_content)

        parsed_tables = []

        for table_match in table_matches:
            # Extract header and body from the table match
            header_row = table_match.group(1).strip()
            body_rows = table_match.group(2).strip().split('\n')

            # Process header and rows to remove leading/trailing pipes
            header_cells = [cell.strip() for cell in header_row.split('|')]
            if header_cells and header_cells[0] == '':
                header_cells = header_cells[1:]
            if header_cells and header_cells[-1] == '':
                header_cells = header_cells[:-1]

            # Process table body rows
            table_data = []
            for row in [header_row] + body_rows:
                if '|' not in row:  # Skip separator row
                    continue

                cells = [cell.strip() for cell in row.split('|')]
                if cells and cells[0] == '':
                    cells = cells[1:]
                if cells and cells[-1] == '':
                    cells = cells[:-1]

                table_data.append(cells)

            # Skip separator row (second row with dashes)
            if len(table_data) > 1 and all('-' in cell for cell in table_data[1]):
                table_data = [table_data[0]] + table_data[2:]

            # Convert to numpy array for easier manipulation
            # Ensure all rows have the same number of columns by padding if necessary
            max_cols = max(len(row) for row in table_data)
            padded_data = [row + [''] * (max_cols - len(row)) for row in table_data]
            table_array = np.array(padded_data)

            parsed_tables.append(table_array)

        return parsed_tables

    def parse_html_tables(self, html_content: str) -> List[np.ndarray]:
        """
        Extract and parse all HTML tables from the provided content.
        
        Args:
            html_content: The HTML content containing tables
            
        Returns:
            A list of numpy arrays, each representing a parsed table
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        parsed_tables = []
        
        for table in tables:
            rows = table.find_all(['tr'])
            table_data = []
            
            for row in rows:
                cells = row.find_all(['th', 'td'])
                row_data = [cell.get_text().strip() for cell in cells]
                table_data.append(row_data)
            
            # Ensure all rows have the same number of columns
            if table_data:
                max_cols = max(len(row) for row in table_data)
                padded_data = [row + [''] * (max_cols - len(row)) for row in table_data]
                table_array = np.array(padded_data)
                parsed_tables.append(table_array)
        
        return parsed_tables

    def run(self, content: str) -> Tuple[bool, str]:
        """
        Run the table test on provided content.
        
        Finds all tables (markdown and/or HTML based on content_type) and checks if any cell 
        matches the target cell and satisfies the specified relationships.
        
        Args:
            content: The content containing tables (markdown or HTML)
            
        Returns:
            A tuple (passed, explanation) where 'passed' is True if the test passes,
            and 'explanation' provides details when the test fails.
        """
        # Initialize variables to track tables and results
        tables_to_check = []
        failed_reasons = []
        
        # Threshold for fuzzy matching derived from max_diffs
        threshold = 1.0 - (self.max_diffs / (len(self.cell) if len(self.cell) > 0 else 1))
        
        # Parse tables based on content_type
        md_tables = self.parse_markdown_tables(content)
        tables_to_check.extend(md_tables)
        
        html_tables = self.parse_html_tables(content)
        tables_to_check.extend(html_tables)
        
        # If no tables found, return failure
        if not tables_to_check:
            return False, f"No tables found in the content"
            
        # Check each table
        for table_array in tables_to_check:
            # Find all cells that match the target cell using fuzzy matching
            matches = []
            for i in range(table_array.shape[0]):
                for j in range(table_array.shape[1]):
                    cell_content = table_array[i, j]
                    similarity = fuzz.ratio(self.cell, cell_content) / 100.0
                    
                    if similarity >= threshold:
                        matches.append((i, j))
            
            # If no matches found in this table, continue to the next table
            if not matches:
                continue
                
            # Check the relationships for each matching cell
            for row_idx, col_idx in matches:
                all_relationships_satisfied = True
                current_failed_reasons = []
                
                # Check up relationship
                if self.up and row_idx > 0:
                    up_cell = table_array[row_idx - 1, col_idx]
                    up_similarity = fuzz.ratio(self.up, up_cell) / 100.0
                    if up_similarity < threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"Cell above '{up_cell}' doesn't match expected '{self.up}' (similarity: {up_similarity:.2f})")
                
                # Check down relationship
                if self.down and row_idx < table_array.shape[0] - 1:
                    down_cell = table_array[row_idx + 1, col_idx]
                    down_similarity = fuzz.ratio(self.down, down_cell) / 100.0
                    if down_similarity < threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"Cell below '{down_cell}' doesn't match expected '{self.down}' (similarity: {down_similarity:.2f})")
                
                # Check left relationship
                if self.left and col_idx > 0:
                    left_cell = table_array[row_idx, col_idx - 1]
                    left_similarity = fuzz.ratio(self.left, left_cell) / 100.0
                    if left_similarity < threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"Cell to the left '{left_cell}' doesn't match expected '{self.left}' (similarity: {left_similarity:.2f})")
                
                # Check right relationship
                if self.right and col_idx < table_array.shape[1] - 1:
                    right_cell = table_array[row_idx, col_idx + 1]
                    right_similarity = fuzz.ratio(self.right, right_cell) / 100.0
                    if right_similarity < threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"Cell to the right '{right_cell}' doesn't match expected '{self.right}' (similarity: {right_similarity:.2f})")
                
                # Check top heading relationship
                if self.top_heading and row_idx > 0:
                    # Find the first non-empty cell in the same column (starting from the top)
                    top_heading_cell = ""
                    for i in range(row_idx):
                        if table_array[i, col_idx].strip():
                            top_heading_cell = table_array[i, col_idx]
                            break
                    
                    if not top_heading_cell:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"No non-empty top heading found in column {col_idx}")
                    else:
                        top_similarity = fuzz.ratio(self.top_heading, top_heading_cell) / 100.0
                        if top_similarity < threshold:
                            all_relationships_satisfied = False
                            current_failed_reasons.append(f"Top heading '{top_heading_cell}' doesn't match expected '{self.top_heading}' (similarity: {top_similarity:.2f})")
                
                # Check left heading relationship
                if self.left_heading and col_idx > 0:
                    # Find the first non-empty cell in the same row (starting from the left)
                    left_heading_cell = ""
                    for j in range(col_idx):
                        if table_array[row_idx, j].strip():
                            left_heading_cell = table_array[row_idx, j]
                            break
                    
                    if not left_heading_cell:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"No non-empty left heading found in row {row_idx}")
                    else:
                        left_heading_similarity = fuzz.ratio(self.left_heading, left_heading_cell) / 100.0
                        if left_heading_similarity < threshold:
                            all_relationships_satisfied = False
                            current_failed_reasons.append(f"Left heading '{left_heading_cell}' doesn't match expected '{self.left_heading}' (similarity: {left_heading_similarity:.2f})")
            
                # If all relationships are satisfied for this cell, the test passes
                if all_relationships_satisfied:
                    return True, ""
                else:
                    failed_reasons.extend(current_failed_reasons)
        
        # If we've gone through all tables and all matching cells and none satisfied all relationships
        if not failed_reasons:
            return False, f"No cell matching '{self.cell}' found in any table with threshold {threshold}"
        else:
            return False, f"Found cells matching '{self.cell}' but relationships were not satisfied: {'; '.join(failed_reasons)}"


@dataclass
class BaselineTest(BasePDFTest):
    """
    This test makes sure that several baseline quality checks pass for the output generation.

    Namely, the output is not blank, not endlessly repeating, and contains characters of the proper
    character sets.

    """
    max_repeats: int=30

    def run(self, content: str) -> Tuple[bool, str]:
        if len("".join(c for c in content if c.isalnum()).strip()) == 0:
            return False, "The text contains no alpha numeric characters"

        # Makes sure that the content has no egregious repeated ngrams at the end, which indicate a degradation of quality
        # Honestly, this test doesn't seem to catch anything at the moment, maybe it can be refactored to a "text-quality"
        # test or something, that measures repetition, non-blanks, charsets, etc
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters(content)
        repeats = d.ngram_repeats()

        for index, count in enumerate(repeats):
            if count > self.max_repeats:
                return False, f"Text ends with {count} repeating {index+1}-grams, invalid"

        pattern = re.compile(
            r'['
            r'\u4e00-\u9FFF'   # CJK Unified Ideographs (Chinese characters)
            r'\u3040-\u309F'   # Hiragana (Japanese)
            r'\u30A0-\u30FF'   # Katakana (Japanese)
            r'\U0001F600-\U0001F64F'  # Emoticons (Emoji)
            r'\U0001F300-\U0001F5FF'  # Miscellaneous Symbols and Pictographs (Emoji)
            r'\U0001F680-\U0001F6FF'  # Transport and Map Symbols (Emoji)
            r'\U0001F1E0-\U0001F1FF'  # Regional Indicator Symbols (flags, Emoji)
            r']',
            flags=re.UNICODE)
        matches = pattern.findall(content)
        if matches:
            return False, f"Text contains disallowed characters {matches}"

        return True, ""


@dataclass
class MathTest(BasePDFTest):
    math: str

    def __post_init__(self):
        super().__post_init__()
        if self.type != TestType.MATH.value:
            raise ValidationError(f"Invalid type for MathTest: {self.type}")
        if len(self.math.strip()) == 0:
            raise ValidationError(f"Math test must have non-empty math expression")

        self.reference_render = render_equation(self.math)

        if self.reference_render is None:
            raise ValidationError(f"Math equation {self.math} was not able to render")


    def run(self, content: str) -> Tuple[bool, str]:
        # Store both the search pattern and the full pattern to replace
        patterns = [
            (r'\$\$(.+?)\$\$', r'\$\$(.+?)\$\$'),        # $$...$$
            (r'\\\((.+?)\\\)', r'\\\((.+?)\\\)'),        # \(...\)
            (r'\\\[(.+?)\\\]', r'\\\[(.+?)\\\]'),        # \[...\]
            (r'\$(.+?)\$', r'\$(.+?)\$')                 # $...$
        ]

        equations = []
        modified_content = content
        
        for search_pattern, replace_pattern in patterns:
            # Find all matches for the current pattern
            matches = re.findall(search_pattern, modified_content, re.DOTALL)
            equations.extend([e.strip() for e in matches])
            
            # Replace all instances of this pattern with empty strings
            modified_content = re.sub(replace_pattern, '', modified_content, flags=re.DOTALL)

        # If an equation in the markdown exactly matches our math string, then that's good enough
        # we don't have to do a more expensive comparison
        if any(hyp == self.math for hyp in equations):
            return True, ""

        # If not, then let's render the math equation itself and now compare to each hypothesis
        best_match_score = 0.0
        best_match_render = None

        for hypothesis in equations:
            hypothesis_render = render_equation(hypothesis)

            if not hypothesis_render:
                continue

            if compare_rendered_equations(self.reference_render, hypothesis_render):
                return True, ""

        # self.reference_render.save(f"maths/{self.id}_ref.png", format="PNG")
        # best_match_render.save(f"maths/{self.id}_hyp.png", format="PNG")

        return False, f"No match found for {self.math} anywhere in content, best match threshold was {best_match_score:.3f}"    



def load_tests(jsonl_file: str) -> List[BasePDFTest]:
    """
    Load tests from a JSONL file.

    Args:
        jsonl_file: Path to the JSONL file containing test definitions.

    Returns:
        A list of test objects.
    """
    tests: List[BasePDFTest] = []
    unique_ids = set()
    with open(jsonl_file, "r") as file:
        for line_number, line in tqdm(enumerate(file, start=1), desc="Loading tests"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                test_type = data.get("type")
                if test_type in {TestType.PRESENT.value, TestType.ABSENT.value}:
                    test = TextPresenceTest(**data)
                elif test_type == TestType.ORDER.value:
                    test = TextOrderTest(**data)
                elif test_type == TestType.TABLE.value:
                    test = TableTest(**data)
                elif test_type == TestType.MATH.value:
                    test = MathTest(**data)
                else:
                    raise ValidationError(f"Unknown test type: {test_type}")

                if test.id in unique_ids:
                    raise ValidationError(f"Test with duplicate id {test.id} found, error loading tests.")
                else:
                    unique_ids.add(test.id)

                tests.append(test)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e}")
                raise
            except (ValidationError, KeyError) as e:
                print(f"Error on line {line_number}: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error on line {line_number}: {e}")
                raise

    return tests


def save_tests(tests: List[BasePDFTest], jsonl_file: str) -> None:
    """
    Save tests to a JSONL file using asdict for conversion.

    Args:
        tests: A list of test objects.
        jsonl_file: Path to the output JSONL file.
    """
    with open(jsonl_file, "w") as file:
        for test in tests:
            file.write(json.dumps(asdict(test)) + "\n")