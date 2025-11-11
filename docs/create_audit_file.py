#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_audit_file.py (robust, prefers risks TABLE if multiple risks sections)

Parses a MkDocs model card Markdown and produces a 3-column auditor spreadsheet:
  High Level | Low Level | Content

- Tolerant to '## 1. Model Overview' etc.
- Handles '**Key:** value' and '**Key**: value'
- Supports bulleted (-, *) and numbered (1.) lists
- Stops subsections at next '###', '##', or '---'
- Accepts '&' or 'and' for 'Compute & Memory'
- UK/US: initialisation/initialization
- **If multiple 'Risks' sections exist**, chooses the one with the risks table
  '| Risk | Description | Stage | Control | Status | Treatment |' (case-insensitive).
  Falls back to the first risks section otherwise.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

SCHEMA = {
    "model overview": [
        "model name", "model version", "parent model", "card author", "contact details",
        "model provider", "model input", "model output", "model task", "model architecture",
        "license", "links and citations"
    ],
    "model usage": [
        "intended use", "out of scope uses", "recommendations for use"
    ],
    "datasets and training": [
        "dataset", "link to dataset card", "dataset purpose", "dataset purpose notes",
        "data pre-processing methods", "model initialisation", "training process"
    ],
    "evaluation": [
        "evaluation overview", "evaluation metric", "metric description",
        "type of evaluation", "evaluation notes"
    ],
    "enabling technology and compute & memory usage": [
        "hardware", "software", "compute and memory"
    ],
    "risks": [
        "risk analysis", "risk", "risk description", "risk treatment stage",
        "type of control required", "risk treatment status", "risk treatment description"
    ]
}

# ---------- basic markdown helpers ----------
def read_markdown(md_path: Path) -> str:
    if not md_path.exists():
        raise FileNotFoundError(f"Input markdown not found: {md_path}")
    return md_path.read_text(encoding="utf-8")

def extract_title(md: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, flags=re.M)
    if m:
        return m.group(1).strip()
    fm = re.search(r"^---\s*(.*?)\s*---", md, flags=re.S)
    if fm:
        t = re.search(r"^\s*title:\s*[\"']?(.+?)[\"']?\s*$", fm.group(1), flags=re.M|re.I)
        if t: return t.group(1).strip()
    return "Model Card"

def strip_front_matter(md: str) -> str:
    return re.sub(r"^---\s*.*?---\s*", "", md, flags=re.S)

def split_sections(md: str) -> Dict[str, str]:
    content = strip_front_matter(md)
    parts = re.split(r"(?m)^##\s+", content)
    sections = {}
    for chunk in parts[1:]:
        header, _, body = chunk.partition("\n")
        sections[header.strip()] = body.strip()
    return sections

def header_matches(header: str, patterns: List[str]) -> bool:
    h = header.lower().strip()
    for pat in patterns:
        if re.search(pat, h):
            return True
    return False

# ---------- map sections (choose right risks section) ----------
def contains_risk_table(text: str) -> bool:
    # Look for a header row with the 6 expected columns (tolerant to spacing/case)
    return re.search(
        r"(?im)^\s*\|\s*risk\s*\|\s*description\s*\|\s*stage\s*\|\s*control\s*\|\s*status\s*\|\s*treatment\s*\|\s*$",
        text
    ) is not None

def map_sections(sections: Dict[str,str]) -> Dict[str,str]:
    mapping: Dict[str,str] = {}

    # straightforward ones (first match wins)
    for hdr in sections.keys():
        if "model overview" not in mapping and header_matches(hdr, [r"^\s*(\d+[\.\)]\s*)?model\s+overview\b"]):
            mapping["model overview"] = hdr
        elif "model usage" not in mapping and header_matches(hdr, [r"^\s*(\d+[\.\)]\s*)?model\s+usage\b"]):
            mapping["model usage"] = hdr
        elif "datasets and training" not in mapping and header_matches(hdr, [r"^\s*(\d+[\.\)]\s*)?(datasets?\s*(and|&)\s*training|data\s*and\s*training)\b"]):
            mapping["datasets and training"] = hdr
        elif "evaluation" not in mapping and header_matches(hdr, [r"^\s*(\d+[\.\)]\s*)?evaluation(\s*$|$)"]) \
             and not header_matches(hdr, [r"protocol", r"mathematical"]):
            mapping["evaluation"] = hdr
        elif "enabling technology and compute & memory usage" not in mapping and header_matches(hdr, [r"^\s*(\d+[\.\)]\s*)?enabling\s+technology.*(compute|memory).*"]):
            mapping["enabling technology and compute & memory usage"] = hdr

    # risks: collect all candidates, then choose the one with a table, else first
    risk_candidates = [hdr for hdr in sections.keys() if header_matches(hdr, [r"^\s*(\d+[\.\)]\s*)?risks?\b"])]
    chosen = None
    for hdr in risk_candidates:
        if contains_risk_table(sections[hdr]):
            chosen = hdr
            break
    if not chosen and risk_candidates:
        chosen = risk_candidates[0]
    if chosen:
        mapping["risks"] = chosen

    return mapping

# ---------- parsing utilities ----------
def subsection(body: str, titles_regex: List[str]) -> str:
    """
    Extract '### Title' subsection text until next '###', '##', or a horizontal rule '---'.
    """
    for t in titles_regex:
        pat = rf"(?mis)^###\s*{t}\s*\n(.*?)(?=^###\s|^##\s|^\s*---\s*$|\Z)"
        m = re.search(pat, body)
        if m:
            return m.group(1).strip()
    return ""

def strip_md_inline(s: str) -> str:
    s = re.sub(r"`([^`]+)`", r"\1", s)    # inline code → text
    return s.strip()

def items_from_block(block: str) -> List[str]:
    items = re.findall(r"(?m)^\s*(?:[-*]|\d+\.)\s+(.*)$", block)
    if items:
        return [strip_md_inline(i.strip()) for i in items]
    block = block.strip()
    return [strip_md_inline(block)] if block else []

def kv_from_bold_lines(body: str) -> Dict[str, str]:
    """
    Support both '**Key:** value' and '**Key**: value'.
    Also collect bulleted block after 'Links & Citations' (colon inside or outside bold).
    """
    out = {}
    pat = re.compile(r"(?:\*\*(.+?)\*\*:\s*(.*)$)|(?:\*\*(.+?):\*\*\s*(.*)$)")
    for line in body.splitlines():
        s = line.strip()
        m = pat.search(s)
        if m:
            key = (m.group(1) or m.group(3)).strip().lower()
            val = strip_md_inline((m.group(2) or m.group(4)).strip())
            out[key] = val
    # capture 'Links &/and Citations', colon inside OR outside bold
    m = re.search(r"(?mis)^\*\*links\s*(?:&|and)\s*citations(?:\*\*:|:\*\*)\s*(.*?)(?=^\s*$|^---|\Z)", body, flags=re.M)
    if m:
        bl = m.group(1)
        items = items_from_block(bl)
        if items:
            out["links and citations"] = " | ".join(items)
    return out

# ---------- harvesters ----------
def harvest_model_overview(text: str) -> Dict[str, List[str]]:
    kv = kv_from_bold_lines(text)
    key_map = {
        "model name":"model name",
        "model version":"model version",
        "parent model":"parent model",
        "card author":"card author",
        "contact":"contact details",
        "contact details":"contact details",
        "model provider":"model provider",
        "model input":"model input",
        "model inputs":"model input",
        "model output":"model output",
        "model outputs":"model output",
        "model task":"model task",
        "model architecture":"model architecture",
        "license":"license",
        "links & citations":"links and citations",
        "links and citations":"links and citations"
    }
    out = {k: [] for k in SCHEMA["model overview"]}
    for k_src, v in kv.items():
        k_norm = key_map.get(k_src, None)
        if k_norm:
            out[k_norm] = [v]
    return out

def harvest_model_usage(text: str) -> Dict[str, List[str]]:
    out = {k: [] for k in SCHEMA["model usage"]}
    intended = subsection(text, [r"Intended\s*Use"])
    oos = subsection(text, [r"Out[-\s]of[-\s]Scope\s*Uses?", r"Out[-\s]of[-\s]Scope"])
    recs = subsection(text, [r"Recommendations\s*for\s*Use", r"Recommendations"])
    out["intended use"] = [" ".join(intended.split())] if intended else []
    oos_list = items_from_block(oos)
    out["out of scope uses"] = ["; ".join(oos_list)] if oos_list else []
    out["recommendations for use"] = [" ".join(recs.split())] if recs else []
    return out

def harvest_datasets_training(text: str) -> Dict[str, List[str]]:
    out = {k: [] for k in SCHEMA["datasets and training"]}
    def grab(lbl):
        # '**Label:** value' OR '**Label**: value'
        m = re.search(rf"\*\*\s*{lbl}\s*:\s*\*\*\s*(.+)$", text, flags=re.I|re.M)
        if m: return strip_md_inline(m.group(1).strip())
        m = re.search(rf"\*\*\s*{lbl}\s*:\s*(.+)$", text, flags=re.I|re.M)
        return strip_md_inline(m.group(1).strip()) if m else ""
    out["dataset"] = [grab("Dataset")] if grab("Dataset") else []
    out["link to dataset card"] = [grab("Link to Dataset Card")] if grab("Link to Dataset Card") else []
    out["dataset purpose"] = [grab("Dataset Purpose")] if grab("Dataset Purpose") else []
    out["dataset purpose notes"] = [grab("Dataset Purpose Notes")] if grab("Dataset Purpose Notes") else []

    dp = subsection(text, [r"Data\s*Pre[-–]processing", r"Data\s*Preprocessing"])
    mi = subsection(text, [r"Model\s*Initiali[sz]ation"])
    tp = subsection(text, [r"Training\s*Process"])

    def join_items(block: str) -> str:
        items = items_from_block(block)
        return "; ".join(items)

    if dp: out["data pre-processing methods"] = [join_items(dp)]
    if mi: out["model initialisation"] = [join_items(mi)]
    if tp: out["training process"] = [join_items(tp)]
    return out

def harvest_evaluation(text: str) -> Dict[str, List[str]]:
    out = {k: [] for k in SCHEMA["evaluation"]}
    evo = subsection(text, [r"Evaluation\s*Overview"])
    evm = subsection(text, [r"Evaluation\s*Metric"])
    md  = subsection(text, [r"Metric\s*Description"])
    toe = subsection(text, [r"Type\s*of\s*Evaluation"])
    env = subsection(text, [r"Evaluation\s*Notes"])
    clean = lambda s: " ".join(s.split()) if s else ""
    out["evaluation overview"] = [clean(evo)] if evo else []
    out["evaluation metric"] = [clean(evm)] if evm else []
    out["metric description"] = [clean(md)] if md else []
    toe_list = items_from_block(toe)
    out["type of evaluation"] = ["; ".join(toe_list)] if toe_list else []
    out["evaluation notes"] = [clean(env)] if env else []
    return out

def harvest_enabling(text: str) -> Dict[str, List[str]]:
    out = {k: [] for k in SCHEMA["enabling technology and compute & memory usage"]}
    m = re.search(r"\*\*\s*Hardware\s*:\s*\*\*\s*(.+)$", text, flags=re.I|re.M)
    if not m: m = re.search(r"\*\*\s*Hardware\s*:\s*(.+)$", text, flags=re.I|re.M)
    if m: out["hardware"] = [" ".join(m.group(1).strip().split())]

    m = re.search(r"\*\*\s*Software\s*:\s*\*\*\s*(.+?)(?:\n\s*\n|$)", text, flags=re.I|re.S)
    if not m: m = re.search(r"\*\*\s*Software\s*:\s*(.+?)(?:\n\s*\n|$)", text, flags=re.I|re.S)
    if m:
        sw_block = m.group(1).strip()
        items = items_from_block(sw_block)
        out["software"] = ["; ".join(items) if items else " ".join(sw_block.split())]

    cm = re.search(r"\*\*\s*Compute\s*(?:&|and)\s*Memory\s*:\s*\*\*(.*?)(?:\n\s*\n|$)", text, flags=re.S|re.I)
    if not cm: cm = re.search(r"\*\*\s*Compute\s*(?:&|and)\s*Memory\s*:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.S|re.I)
    if cm:
        block = cm.group(1)
        items = items_from_block(block)
        out["compute and memory"] = ["; ".join(items) if items else " ".join(block.split())]
    return out

def harvest_risks(text: str) -> Tuple[str, List[Dict[str,str]]]:
    pre = re.split(r"(?m)^\|", text, maxsplit=1)[0].strip()
    risk_analysis = " ".join(pre.split()) if pre else ""
    rows = []
    # Table: header row + separator + body lines
    table = re.search(r"(?mis)^\s*\|([^|]*\|){5}.*\n\s*\|[-:\s\|]+\|\s*\n(.*?)(?:\n\s*\n|\Z)", text)
    if table:
        body = table.group(2).strip()
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 6:
                rows.append({
                    "risk": cols[0],
                    "risk description": cols[1],
                    "risk treatment stage": cols[2],
                    "type of control required": cols[3],
                    "risk treatment status": cols[4],
                    "risk treatment description": cols[5],
                })
    return risk_analysis, rows

# ---------- orchestrate ----------
def normalize_to_rows(md_text: str, include_missing: bool=True) -> List[Tuple[str,str,str]]:
    sections = split_sections(md_text)
    secmap = map_sections(sections)
    rows: List[Tuple[str,str,str]] = []

    def emit(section_key: str, data: Dict[str, List[str]]):
        for low in SCHEMA[section_key]:
            vals = data.get(low, [])
            if vals:
                rows.append((section_key, low, " | ".join(vals)))
            elif include_missing:
                rows.append((section_key, low, "TBD"))

    if "model overview" in secmap:
        data = harvest_model_overview(sections[secmap["model overview"]])
        emit("model overview", data)
    if "model usage" in secmap:
        data = harvest_model_usage(sections[secmap["model usage"]])
        emit("model usage", data)
    if "datasets and training" in secmap:
        data = harvest_datasets_training(sections[secmap["datasets and training"]])
        emit("datasets and training", data)
    if "evaluation" in secmap:
        data = harvest_evaluation(sections[secmap["evaluation"]])
        emit("evaluation", data)
    if "enabling technology and compute & memory usage" in secmap:
        data = harvest_enabling(sections[secmap["enabling technology and compute & memory usage"]])
        emit("enabling technology and compute & memory usage", data)
    if "risks" in secmap:
        analysis, risk_rows = harvest_risks(sections[secmap["risks"]])
        rows.append(("risks", "risk analysis", analysis if analysis else ("TBD" if include_missing else "")))
        if risk_rows:
            for r in risk_rows:
                for k in ["risk","risk description","risk treatment stage",
                          "type of control required","risk treatment status","risk treatment description"]:
                    rows.append(("risks", k, r.get(k, "TBD" if include_missing else "")))
        elif include_missing:
            for k in ["risk","risk description","risk treatment stage",
                      "type of control required","risk treatment status","risk treatment description"]:
                rows.append(("risks", k, "TBD"))
    return rows

def rows_to_dataframe(rows: List[Tuple[str,str,str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["High Level","Low Level","Content"])
    order = list(SCHEMA.keys())
    order_map = {name:i for i,name in enumerate(order)}
    df["__o"] = df["High Level"].map(order_map)
    df = df.sort_values(["__o","Low Level"]).drop(columns="__o").reset_index(drop=True)
    return df

def write_excel(df: pd.DataFrame, out_xlsx: Path, title: str):
    # choose engine
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            raise RuntimeError("Install an Excel engine: pip install xlsxwriter or pip install openpyxl")

    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
        startrow = 2
        df.to_excel(writer, index=False, sheet_name="Auditor View", startrow=startrow)
        if engine == "xlsxwriter":
            wb = writer.book
            ws = writer.sheets["Auditor View"]
            title_fmt = wb.add_format({"bold": True, "font_size": 16, "align": "left", "valign": "vcenter"})
            ws.merge_range(0, 0, 0, 2, f"Auditor Model Card — {title}", title_fmt)
            header_fmt = wb.add_format({"bold": True, "bg_color": "#E8EEF6", "border": 1})
            for c, name in enumerate(df.columns):
                ws.write(startrow, c, name, header_fmt)
            wrap = wb.add_format({"text_wrap": True, "valign": "top"})
            ws.set_column(0, 0, 38, wrap)
            ws.set_column(1, 1, 40, wrap)
            ws.set_column(2, 2, 100, wrap)
            ws.autofilter(startrow, 0, startrow + len(df), 2)
            ws.freeze_panes(startrow+1, 0)

def main():
    ap = argparse.ArgumentParser(description="MkDocs model card → auditor spreadsheet (High Level | Low Level | Content)")
    ap.add_argument("--input","-i", required=True, type=Path, help="Path to model card Markdown (.md)")
    ap.add_argument("--output","-o", required=True, type=Path, help="Path to output Excel (.xlsx)")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    ap.add_argument("--no-missing", action="store_true", help="Do not include missing fields as 'TBD'")
    args = ap.parse_args()

    md = read_markdown(args.input)
    title = extract_title(md)
    rows = normalize_to_rows(md, include_missing=not args.no_missing)
    df = rows_to_dataframe(rows)
    write_excel(df, args.output, title)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(f"[OK] Wrote {len(df)} rows → {args.output}")
    if args.csv:
        print(f"[OK] CSV → {args.csv}")

if __name__ == "__main__":
    main()

