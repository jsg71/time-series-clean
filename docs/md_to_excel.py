#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
md_to_audit_sheet.py

- Reads one or more markdown model cards.
- Each card goes to its own sheet in a single Excel workbook.
- Sheets have columns:

    Section | Exploration Area | Ref | Field Name | Prompt | Response Area

Layout per sheet:
- Row 1 : merged title "Model Cards Plus - Model Card" (blue, bold)
- Row 2 : header row (yellow)
- Row 3+: data rows in top-to-bottom order from markdown.

For each contiguous block with the same (Section, Exploration Area),
the cells in the first two columns are vertically merged,
centered, and wrapped so the text is readable.

Tables:
- Markdown tables are kept as multi-line text in the Response Area cell.
- The whole Response Area column is set to wrap text so table rows are readable
  within a single Excel cell (one row per M-block).

USAGE EXAMPLES
--------------
Single file:
    python md_to_audit_sheet.py -i model_cards/example1.md -o cards.xlsx

Folder of cards (recommended):
    python md_to_audit_sheet.py -d model_cards -o cards.xlsx
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd


# --------------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------------

# Exploration Area text per section (per *normalized* section name).
SECTION_EXPLORATION: Dict[str, str] = {
    "Model Overview": "High-level exploration of what the model is, its purpose, and context.",
    "Model Usage": "Exploration of how the model is used, intended scope, and out-of-scope uses.",
    "Data": "Exploration of input data sources, quality, and representativeness.",
    "Training": "Exploration of training procedure, hyperparameters, and infrastructure.",
    "Evaluation": "Exploration of model evaluation, metrics, and performance.",
    "Monitoring": "Exploration of live monitoring, alerts, and retraining triggers.",
    "Risks": "Exploration of model risks, limitations, and mitigations.",
    "Governance": "Exploration of approvals, versioning, and audit trail.",
    # Extend or customise as needed.
}

# Mapping from M-codes to prompts that you want in the spreadsheet.
# If an M-code is not listed here, Prompt will just be empty (you can fill it later).
M_PROMPTS: Dict[str, str] = {
    "M1": "Summarise the overall purpose of the model.",
    "M2": "Who are the primary stakeholders or audiences?",
    "M3": "List known limitations and weaknesses.",
    "M4": "Describe the intended, approved uses of the model.",
    "M5": "Describe out-of-scope or prohibited uses.",
    "M6": "Summarise evaluation metrics and headline performance.",
    "M7": "Summarise any fairness or bias analysis.",
    "M8": "Provide a high-level summary of key risks.",
    "M9": "Detail specific risks, their status and mitigations.",
    "M10": "Describe the data sources and key preprocessing steps.",
    "M11": "Describe how training was performed, including objective and key hyperparameters.",
    "M12": "Explain the monitoring strategy and alert thresholds.",
    "M13": "Document governance, approvals, and sign-off responsibilities.",
    "M14": "Describe data representativeness and coverage gaps.",
    "M15": "Describe privacy and security controls applied to the data.",
    "M16": "Explain how model updates and rollbacks are managed.",
    "M17": "Summarise interpretability/explainability techniques used.",
    "M18": "Summarise stress testing or scenario analysis that has been performed.",
    "M19": "Describe any user override / human-in-the-loop mechanisms.",
    "M20": "Describe key open issues and planned future work.",
}


# --------------------------------------------------------------------
#  MARKDOWN PARSING UTILITIES
# --------------------------------------------------------------------

def read_markdown(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    return path.read_text(encoding="utf-8")


def split_into_sections(md: str) -> List[Tuple[str, str]]:
    """
    Split markdown into (section_header, section_body) based on '## ' headings.

    Returns: list of (raw_header_text, section_body).
    Example header text: '1. Model Overview'
    """
    # Remove YAML front-matter if present
    md = re.sub(r"^---\s*.*?---\s*", "", md, flags=re.S)

    # Split on "## " headings
    parts = re.split(r"(?m)^##\s+", md)

    sections: List[Tuple[str, str]] = []
    # parts[0] is content before first '##' (e.g. '# Title'); ignore for now
    for chunk in parts[1:]:
        header_line, _, body = chunk.partition("\n")
        header_line = header_line.strip()
        sections.append((header_line, body.strip()))
    return sections


def normalize_section_name(raw_header: str) -> str:
    """
    Turn '1. Model Overview' -> 'Model Overview'
         '2) Model Usage'   -> 'Model Usage'
    If no leading numbering, return as-is.
    """
    cleaned = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", raw_header).strip()
    return cleaned


def parse_m_blocks(section_body: str) -> List[Tuple[str, str, str]]:
    """
    Given the body text of a section, find all M-blocks.

    Format:

        M1: Field name
        <content until next Mx: or end-of-section>

    Returns list of (M_code, field_name, block_text):
        M_code      -> 'M1', 'M2', ...
        field_name  -> text after 'M1:'
        block_text  -> multiline text after that line
    """
    lines = section_body.splitlines()

    blocks: List[Tuple[str, str, str]] = []
    current_m: Optional[str] = None
    current_field_name: str = ""
    current_lines: List[str] = []

    # Regex for "M12: some field name"
    m_re = re.compile(r"^\s*(M\d+)\s*:\s*(.*)$")

    for line in lines:
        m = m_re.match(line)
        if m:
            # Start of a new M-block
            if current_m is not None:
                blocks.append((current_m, current_field_name,
                               "\n".join(current_lines).strip()))
                current_lines = []

            current_m = m.group(1)
            current_field_name = m.group(2).strip() if m.group(2) else ""
        else:
            if current_m is not None:
                current_lines.append(line)

    # Flush last block
    if current_m is not None:
        blocks.append((current_m, current_field_name,
                       "\n".join(current_lines).strip()))

    return blocks


# --------------------------------------------------------------------
#  MARKDOWN CLEANING / LATEX HANDLING
# --------------------------------------------------------------------

def strip_basic_markdown(text: str, keep_tables: bool = True) -> str:
    """
    Simple markdown "cleaner" to make text easier to read in Excel.

    - Optionally keeps table rows (lines starting with '|') as-is.
    - Removes headings, bullet markers, bold/italic markers, and code fences.
    - Converts links [text](url) -> 'text'.
    - Leaves LaTeX content to handle_latex().
    """
    result_lines: List[str] = []
    in_code_block = False

    for line in text.splitlines():
        stripped = line.rstrip("\n")

        # Handle fenced code blocks
        if stripped.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            result_lines.append(stripped)
            continue

        # Keep tables untouched
        if keep_tables and stripped.lstrip().startswith("|"):
            result_lines.append(stripped)
            continue

        # Drop headings
        if re.match(r"^\s*#{1,6}\s+", stripped):
            continue

        # Remove bullet / numbered markers
        stripped = re.sub(r"^\s*[-*]\s+", "", stripped)
        stripped = re.sub(r"^\s*\d+\.\s+", "", stripped)

        # Convert [text](url) -> text
        stripped = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", stripped)

        # Inline code `code` -> code
        stripped = re.sub(r"`([^`]+)`", r"\1", stripped)

        # Strip bold/italic markers
        stripped = stripped.replace("**", "").replace("__", "")
        stripped = stripped.replace("*", "").replace("_", "")

        if stripped.strip():
            result_lines.append(stripped)

    return "\n".join(result_lines).strip()


def handle_latex(text: str, mode: str = "strip_delimiters") -> str:
    """
    Basic options for LaTeX formulae:

    mode = "keep"             -> keep $...$ and $$...$$
    mode = "strip_delimiters" -> remove $ / $$ but keep TeX
    mode = "remove"           -> remove formulae entirely
    """
    if mode == "keep":
        return text

    if mode == "strip_delimiters":
        text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.S)
        text = re.sub(r"\$(.+?)\$", r"\1", text, flags=re.S)
        return text

    if mode == "remove":
        text = re.sub(r"\$\$(.+?)\$\$", "", text, flags=re.S)
        text = re.sub(r"\$(.+?)\$", "", text, flags=re.S)
        return text

    return text


def clean_for_excel(raw_block: str) -> str:
    """
    High-level cleaning pipeline for text going into 'Response Area'.
    """
    text = strip_basic_markdown(raw_block, keep_tables=True)
    text = handle_latex(text, mode="strip_delimiters")
    return text


# --------------------------------------------------------------------
#  MAIN: MARKDOWN -> ROWS -> DATAFRAME
# --------------------------------------------------------------------

def md_to_rows(md_text: str) -> List[Dict[str, str]]:
    """
    Convert markdown into list-of-row dicts.

    Each row:
        Section          (e.g. 'Model Overview')
        Exploration Area (from SECTION_EXPLORATION)
        Ref              (e.g. 'M1')
        Field Name       (on 'M1:' line)
        Prompt           (from M_PROMPTS)
        Response Area    (block content after that line)
    """
    sections = split_into_sections(md_text)
    rows: List[Dict[str, str]] = []

    for raw_header, body in sections:
        section_name = normalize_section_name(raw_header)
        exploration_area = SECTION_EXPLORATION.get(section_name, "")

        for m_code, field_name, raw_block in parse_m_blocks(body):
            prompt = M_PROMPTS.get(m_code, "")
            cleaned_text = clean_for_excel(raw_block)

            rows.append(
                {
                    "Section": section_name,
                    "Exploration Area": exploration_area,
                    "Ref": m_code,
                    "Field Name": field_name,
                    "Prompt": prompt,
                    "Response Area": cleaned_text,
                }
            )

    return rows


def rows_to_df(rows: List[Dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "Section",
            "Exploration Area",
            "Ref",
            "Field Name",
            "Prompt",
            "Response Area",
        ],
    )


# --------------------------------------------------------------------
#  EXCEL WRITING (MULTI-SHEET, MERGED SECTION CELLS)
# --------------------------------------------------------------------

def sanitize_sheet_name(name: str) -> str:
    """
    Make a filesystem filename into a valid Excel sheet name.
    """
    # Max 31 chars, no []:*?/\
    safe = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return safe[:31] or "Sheet1"


def format_sheet_xlsxwriter(workbook, worksheet, df: pd.DataFrame):
    """
    Apply formatting / merging to a sheet that already has the df written,
    starting at row 2 (0-based).
    """
    num_cols = len(df.columns)

    # Formats
    title_format = workbook.add_format(
        {"bold": True, "font_color": "blue", "font_size": 14}
    )
    header_format = workbook.add_format(
        {"bold": True, "bg_color": "yellow", "border": 1}
    )
    merged_header_format = workbook.add_format(
        {
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 1,
        }
    )
    # Body format for Response Area: wrap, top-aligned
    response_wrap_format = workbook.add_format(
        {
            "text_wrap": True,
            "valign": "top",
        }
    )

    # Title row
    worksheet.merge_range(
        0, 0, 0, num_cols - 1, "Model Cards Plus - Model Card", title_format
    )

    # Header row (our only header row)
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(1, col_idx, col_name, header_format)

    # Column widths (Response Area gets wrap format)
    worksheet.set_column(0, 0, 18)                             # Section
    worksheet.set_column(1, 1, 45)                             # Exploration Area
    worksheet.set_column(2, 2, 8)                              # Ref
    worksheet.set_column(3, 3, 35)                             # Field Name
    worksheet.set_column(4, 4, 40)                             # Prompt
    worksheet.set_column(5, 5, 80, response_wrap_format)       # Response Area (wrapped)

    # Merge Section / Exploration Area for contiguous groups
    if df.empty:
        return

    excel_data_start_row = 2  # df index 0 is Excel row 2 (0-based)

    group_start = 0
    prev_section = str(df.loc[0, "Section"]).strip()
    prev_expl = str(df.loc[0, "Exploration Area"]).strip()
    n = len(df)

    for i in range(1, n + 1):  # go one past end to flush last group
        flush = False
        if i == n:
            flush = True
            end_idx = n - 1
        else:
            sec = str(df.loc[i, "Section"]).strip()
            expl = str(df.loc[i, "Exploration Area"]).strip()
            if sec != prev_section or expl != prev_expl:
                flush = True
                end_idx = i - 1

        if flush:
            row_start = excel_data_start_row + group_start
            row_end = excel_data_start_row + end_idx

            # set a decent height so wrapped text is visible
            for r in range(row_start, row_end + 1):
                worksheet.set_row(r, 42)

            if row_start == row_end:
                # Single-row group: just write formatted
                worksheet.write(row_start, 0, prev_section, merged_header_format)
                worksheet.write(row_start, 1, prev_expl, merged_header_format)
            else:
                worksheet.merge_range(
                    row_start, 0, row_end, 0, prev_section, merged_header_format
                )
                worksheet.merge_range(
                    row_start, 1, row_end, 1, prev_expl, merged_header_format
                )

            if i < n:
                group_start = i
                prev_section = str(df.loc[i, "Section"]).strip()
                prev_expl = str(df.loc[i, "Exploration Area"]).strip()


def write_excel_multi(dfs: Dict[str, pd.DataFrame], out_path: Path):
    """
    Write multiple DataFrames to a single Excel workbook, one sheet per DF.
    """
    if not dfs:
        raise ValueError("No dataframes to write.")

    # Choose engine
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            raise RuntimeError(
                "Please install an Excel engine, e.g.:\n"
                "  pip install xlsxwriter\n"
                "or\n"
                "  pip install openpyxl"
            )

    with pd.ExcelWriter(out_path, engine=engine) as writer:
        workbook = writer.book

        for raw_name, df in dfs.items():
            sheet_name = sanitize_sheet_name(raw_name)
            # header=False so we don't get a duplicate header row
            df.to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
                startrow=2,
                header=False,
            )
            worksheet = writer.sheets[sheet_name]

            if engine == "xlsxwriter":
                format_sheet_xlsxwriter(workbook, worksheet, df)
            else:
                # Minimal formatting for openpyxl
                worksheet.cell(row=1, column=1, value="Model Cards Plus - Model Card")

    print(f"Wrote Excel workbook: {out_path} with {len(dfs)} sheet(s)")


# --------------------------------------------------------------------
#  CLI
# --------------------------------------------------------------------

def load_markdown_sources(
    input_file: Optional[Path], input_dir: Optional[Path]
) -> Dict[str, str]:
    """
    Return mapping: sheet_base_name -> markdown_text.
    - If input_dir is given, load all *.md files in that directory.
    - Else, use input_file.
    """
    sources: Dict[str, str] = {}

    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        for md_path in sorted(input_dir.glob("*.md")):
            sources[md_path.stem] = read_markdown(md_path)
    elif input_file is not None:
        sources[input_file.stem] = read_markdown(input_file)
    else:
        raise ValueError("Provide either --input or --input_dir")

    if not sources:
        raise ValueError("No markdown files found.")
    return sources


def main():
    ap = argparse.ArgumentParser(
        description="Convert one or more markdown model cards into an Excel workbook."
    )
    ap.add_argument(
        "-i", "--input", type=Path, help="Single input markdown file (.md)"
    )
    ap.add_argument(
        "-d",
        "--input_dir",
        type=Path,
        help="Directory containing multiple markdown model cards (.md)",
    )
    ap.add_argument(
        "-o", "--output", required=True, type=Path, help="Output Excel file (.xlsx)"
    )
    args = ap.parse_args()

    sources = load_markdown_sources(args.input, args.input_dir)

    dfs: Dict[str, pd.DataFrame] = {}
    for name, md_text in sources.items():
        rows = md_to_rows(md_text)
        if not rows:
            print(f"WARNING: no M-blocks found in {name}, skipping.")
            continue
        dfs[name] = rows_to_df(rows)

    if not dfs:
        print("No valid sheets to write; exiting.")
        return

    write_excel_multi(dfs, args.output)


if __name__ == "__main__":
    main()

#python md_to_excel.py -d example_cards -o model_cards_plus.xlsx

