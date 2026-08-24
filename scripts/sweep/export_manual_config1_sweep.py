#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import re

import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


LOG_PATTERNS = {
    "request_throughput": re.compile(r"Request throughput \(req/s\):\s*([0-9.]+)"),
    "output_token_throughput": re.compile(r"Output token throughput \(tok/s\):\s*([0-9.]+)"),
    "mean_ttft_ms": re.compile(r"Mean TTFT \(ms\):\s*([0-9.]+)"),
    "mean_tpot_ms": re.compile(r"Mean TPOT \(ms\):\s*([0-9.]+)"),
    "successful_requests": re.compile(r"Successful requests:\s*([0-9]+)"),
    "failed_requests": re.compile(r"Failed requests:\s*([0-9]+)"),
}

FINAL_LOG_RE = re.compile(r"config1_c(?P<concurrency>[0-9]+)\.log$")
ATTEMPT_LOG_RE = re.compile(r"config1_c(?P<concurrency>[0-9]+)_attempt(?P<attempt>[0-9]+)\.log$")

SUMMARY_COLUMNS = [
    "concurrency",
    "request_throughput",
    "output_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "attempt_count",
    "final_success",
    "successful_requests",
    "failed_requests",
    "final_log",
]

ATTEMPT_COLUMNS = [
    "concurrency",
    "attempt",
    "success",
    "successful_requests",
    "failed_requests",
    "request_throughput",
    "output_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "log_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export manual_config1_sweep results to XLSX and trend charts.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input directories, log files, or summary.csv. Defaults to the current directory.",
    )
    parser.add_argument(
        "--xlsx",
        help="Output XLSX file path. Defaults to <input_dir>/manual_config1_sweep.xlsx.",
    )
    parser.add_argument(
        "--chart-dir",
        help="Directory for SVG charts. Defaults to <input_dir>/charts.",
    )
    parser.add_argument(
        "--summary-out",
        help="Output CSV path for enriched summary. Defaults to <input_dir>/summary_enriched.csv.",
    )
    return parser.parse_args()


def iter_log_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("config1_c*.log") if path.is_file())


def resolve_inputs(inputs: list[str]) -> tuple[Path, Path, list[Path]]:
    raw_inputs = inputs or ["."]
    resolved_inputs = [Path(value).expanduser().resolve() for value in raw_inputs]

    missing_paths = [path for path in resolved_inputs if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise SystemExit(f"Input path does not exist: {missing_text}")

    if len(resolved_inputs) == 1 and resolved_inputs[0].is_dir():
        input_dir = resolved_inputs[0]
        return input_dir, input_dir / "summary.csv", iter_log_files(input_dir)

    log_paths: list[Path] = []
    summary_csv: Path | None = None
    output_candidates: list[Path] = []

    for path in resolved_inputs:
        if path.is_dir():
            output_candidates.append(path)
            if summary_csv is None:
                candidate = path / "summary.csv"
                if candidate.exists():
                    summary_csv = candidate
            log_paths.extend(iter_log_files(path))
            continue

        output_candidates.append(path.parent)
        if path.name == "summary.csv":
            summary_csv = path
            continue
        log_paths.append(path)

    if not output_candidates:
        output_dir = Path.cwd().resolve()
    else:
        common_parent = Path(os.path.commonpath([str(path) for path in output_candidates]))
        output_dir = common_parent if common_parent.is_dir() else Path.cwd().resolve()

    if summary_csv is None:
        summary_csv = output_dir / "summary.csv"

    unique_logs: dict[Path, None] = {}
    for path in log_paths:
        unique_logs[path] = None

    return output_dir, summary_csv, sorted(unique_logs)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_metrics(text: str) -> dict[str, float | int | bool | None]:
    metrics: dict[str, float | int | bool | None] = {}
    for key, pattern in LOG_PATTERNS.items():
        match = pattern.search(text)
        if match:
            raw_value = match.group(1)
            if key.endswith("requests"):
                metrics[key] = int(raw_value)
            else:
                metrics[key] = float(raw_value)
        else:
            metrics[key] = None

    metrics["success"] = (
        metrics["failed_requests"] == 0
        and (metrics["successful_requests"] or 0) > 0
    )
    return metrics


def read_summary_csv(path: Path) -> dict[int, dict[str, float | int | str | None]]:
    rows: dict[int, dict[str, float | int | str | None]] = {}
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            concurrency = parse_int(row.get("concurrency"))
            if concurrency is None:
                continue
            rows[concurrency] = {
                "concurrency": concurrency,
                "request_throughput": parse_float(row.get("request_throughput")),
                "output_token_throughput": parse_float(row.get("output_token_throughput")),
                "mean_ttft_ms": parse_float(row.get("mean_ttft_ms")),
                "mean_tpot_ms": parse_float(row.get("mean_tpot_ms")),
            }
    return rows


def collect_logs(log_paths: Iterable[Path]) -> tuple[dict[int, dict], list[dict]]:
    final_logs: dict[int, dict] = {}
    attempts: list[dict] = []

    for path in sorted(log_paths):
        attempt_match = ATTEMPT_LOG_RE.match(path.name)
        final_match = FINAL_LOG_RE.match(path.name)
        if not attempt_match and not final_match:
            continue
        text = read_text(path)
        metrics = extract_metrics(text)

        if attempt_match:
            attempts.append(
                {
                    "concurrency": int(attempt_match.group("concurrency")),
                    "attempt": int(attempt_match.group("attempt")),
                    "log_file": path.name,
                    **metrics,
                }
            )
            continue

        if final_match:
            concurrency = int(final_match.group("concurrency"))
            final_logs[concurrency] = {
                "concurrency": concurrency,
                "log_file": path.name,
                **metrics,
            }

    attempts.sort(key=lambda row: (row["concurrency"], row["attempt"]))
    return final_logs, attempts


def build_summary_rows(
    summary_csv_rows: dict[int, dict[str, float | int | str | None]],
    final_logs: dict[int, dict],
    attempts: list[dict],
) -> list[dict[str, float | int | str | bool | None]]:
    attempts_by_concurrency: dict[int, list[dict]] = defaultdict(list)
    for row in attempts:
        attempts_by_concurrency[row["concurrency"]].append(row)

    all_concurrencies = sorted(
        set(summary_csv_rows.keys())
        | set(final_logs.keys())
        | set(attempts_by_concurrency.keys())
    )

    summary_rows: list[dict[str, float | int | str | bool | None]] = []
    for concurrency in all_concurrencies:
        csv_row = summary_csv_rows.get(concurrency, {})
        final_row = final_logs.get(concurrency, {})
        attempt_rows = attempts_by_concurrency.get(concurrency, [])

        summary_rows.append(
            {
                "concurrency": concurrency,
                "request_throughput": csv_row.get("request_throughput", final_row.get("request_throughput")),
                "output_token_throughput": csv_row.get(
                    "output_token_throughput",
                    final_row.get("output_token_throughput"),
                ),
                "mean_ttft_ms": csv_row.get("mean_ttft_ms", final_row.get("mean_ttft_ms")),
                "mean_tpot_ms": csv_row.get("mean_tpot_ms", final_row.get("mean_tpot_ms")),
                "attempt_count": len(attempt_rows),
                "final_success": final_row.get("success"),
                "successful_requests": final_row.get("successful_requests"),
                "failed_requests": final_row.get("failed_requests"),
                "final_log": final_row.get("log_file"),
            }
        )
    return summary_rows


def write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def xml_cell(value: object) -> str:
    if value is None:
        return "<c/>"
    if isinstance(value, bool):
        return f"<c t=\"inlineStr\"><is><t>{'TRUE' if value else 'FALSE'}</t></is></c>"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "<c/>"
        return f"<c><v>{value}</v></c>"
    return f"<c t=\"inlineStr\"><is><t>{escape(str(value))}</t></is></c>"


def build_sheet_xml(rows: list[dict], columns: list[str]) -> str:
    sheet_rows = []
    header_cells = "".join(xml_cell(column) for column in columns)
    sheet_rows.append(f"<row r=\"1\">{header_cells}</row>")
    for index, row in enumerate(rows, start=2):
        row_cells = "".join(xml_cell(row.get(column)) for column in columns)
        sheet_rows.append(f"<row r=\"{index}\">{row_cells}</row>")
    sheet_data = "".join(sheet_rows)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        f"<sheetData>{sheet_data}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[str], list[dict]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    content_types = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
        "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>",
        "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>",
        "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>",
        "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>",
    ]
    for index in range(1, len(sheets) + 1):
        content_types.append(
            f"<Override PartName=\"/xl/worksheets/sheet{index}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = []
    for index, (name, _, _) in enumerate(sheets, start=1):
        workbook_sheets.append(
            f"<sheet name=\"{escape(name)}\" sheetId=\"{index}\" r:id=\"rId{index}\"/>"
        )
        workbook_rels.append(
            f"<Relationship Id=\"rId{index}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet{index}.xml\"/>"
        )
    workbook_rels.append(
        f"<Relationship Id=\"rId{len(sheets) + 1}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr(
            "_rels/.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
            "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
            "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/>"
            "</Relationships>",
        )
        archive.writestr(
            "docProps/app.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
            "<Application>GitHub Copilot</Application>"
            f"<TitlesOfParts><vt:vector size=\"{len(sheets)}\" baseType=\"lpstr\">"
            + "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name, _, _ in sheets)
            + "</vt:vector></TitlesOfParts>"
            f"<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(sheets)}</vt:i4></vt:variant></vt:vector></HeadingPairs>"
            "</Properties>",
        )
        archive.writestr(
            "docProps/core.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
            "<dc:creator>GitHub Copilot</dc:creator>"
            "<cp:lastModifiedBy>GitHub Copilot</cp:lastModifiedBy>"
            "<dc:title>manual_config1_sweep export</dc:title>"
            f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:created>"
            f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:modified>"
            "</cp:coreProperties>",
        )
        archive.writestr(
            "xl/workbook.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            f"<sheets>{''.join(workbook_sheets)}</sheets>"
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            f"{''.join(workbook_rels)}"
            "</Relationships>",
        )
        archive.writestr(
            "xl/styles.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
            "<fills count=\"2\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill></fills>"
            "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
            "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
            "<cellXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/></cellXfs>"
            "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
            "</styleSheet>",
        )
        for index, (_, columns, rows) in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                build_sheet_xml(rows, columns),
            )


def format_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    point_values = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2.5\" points=\"{point_values}\" />"


def svg_circles(points: list[tuple[float, float]], color: str) -> str:
    return "".join(
        f"<circle cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"4\" fill=\"{color}\" />"
        for x, y in points
    )


def write_svg_chart(
    output_path: Path,
    rows: list[dict[str, float | int | str | bool | None]],
    metric_key: str,
    title: str,
    y_label: str,
) -> None:
    data = [
        (int(row["concurrency"]), float(row[metric_key]))
        for row in rows
        if row.get("concurrency") is not None and row.get(metric_key) is not None
    ]
    if not data:
        return

    data.sort(key=lambda item: item[0])
    width = 960
    height = 540
    left = 90
    right = 40
    top = 60
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    xs = [item[0] for item in data]
    ys = [item[1] for item in data]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        padding = max(min_y * 0.1, 1.0)
        min_y -= padding
        max_y += padding
    else:
        padding = (max_y - min_y) * 0.1
        min_y -= padding
        max_y += padding

    def scale_x(value: float) -> float:
        return left + ((value - min_x) / (max_x - min_x)) * plot_width

    def scale_y(value: float) -> float:
        return top + (1 - ((value - min_y) / (max_y - min_y))) * plot_height

    points = [(scale_x(x), scale_y(y)) for x, y in data]
    y_ticks = 5
    tick_markup = []
    label_markup = []
    for index in range(y_ticks + 1):
        value = min_y + ((max_y - min_y) * index / y_ticks)
        y = scale_y(value)
        tick_markup.append(
            f"<line x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{width - right}\" y2=\"{y:.2f}\" stroke=\"#d7dde5\" stroke-width=\"1\" />"
        )
        label_markup.append(
            f"<text x=\"{left - 12}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-size=\"12\" fill=\"#44566c\">{escape(format_metric(value))}</text>"
        )

    for x_value in xs:
        x = scale_x(x_value)
        tick_markup.append(
            f"<line x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{height - bottom}\" stroke=\"#eef2f6\" stroke-width=\"1\" />"
        )
        label_markup.append(
            f"<text x=\"{x:.2f}\" y=\"{height - bottom + 24}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#44566c\">{x_value}</text>"
        )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfcfe" />
  <text x="{width / 2:.0f}" y="30" text-anchor="middle" font-size="24" font-family="Segoe UI, Arial, sans-serif" fill="#132238">{escape(title)}</text>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#44566c">Concurrency</text>
  <text x="24" y="{height / 2:.0f}" text-anchor="middle" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#44566c" transform="rotate(-90 24 {height / 2:.0f})">{escape(y_label)}</text>
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#c5d0dc" stroke-width="1.2" rx="8" />
  {''.join(tick_markup)}
  {''.join(label_markup)}
  {svg_polyline(points, '#0b84f3')}
  {svg_circles(points, '#0b84f3')}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir, summary_csv, log_paths = resolve_inputs(args.inputs)
    xlsx_path = Path(args.xlsx).expanduser().resolve() if args.xlsx else input_dir / "manual_config1_sweep.xlsx"
    chart_dir = Path(args.chart_dir).expanduser().resolve() if args.chart_dir else input_dir / "charts"
    summary_out = Path(args.summary_out).expanduser().resolve() if args.summary_out else input_dir / "summary_enriched.csv"

    summary_rows_csv = read_summary_csv(summary_csv)
    final_logs, attempts = collect_logs(log_paths)
    summary_rows = build_summary_rows(summary_rows_csv, final_logs, attempts)

    if not summary_rows:
        raise SystemExit(f"No summary rows or config1 logs were found under: {input_dir}")

    write_csv(summary_out, SUMMARY_COLUMNS, summary_rows)
    write_xlsx(
        xlsx_path,
        [
            ("summary", SUMMARY_COLUMNS, summary_rows),
            ("attempts", ATTEMPT_COLUMNS, attempts),
        ],
    )

    chart_specs = [
        ("request_throughput", "Request Throughput vs Concurrency", "req/s"),
        ("output_token_throughput", "Output Token Throughput vs Concurrency", "tok/s"),
        ("mean_ttft_ms", "Mean TTFT vs Concurrency", "ms"),
        ("mean_tpot_ms", "Mean TPOT vs Concurrency", "ms"),
    ]
    for metric_key, title, y_label in chart_specs:
        write_svg_chart(chart_dir / f"{metric_key}.svg", summary_rows, metric_key, title, y_label)

    print(f"Input directory: {input_dir}")
    print(f"Excel workbook: {xlsx_path}")
    print(f"Enriched summary: {summary_out}")
    print(f"Chart directory: {chart_dir}")


if __name__ == "__main__":
    main()