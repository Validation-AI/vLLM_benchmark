import sys, csv, html

path = sys.argv[1]

# Columns we want to show, mapped from possible header names in summary log.
DISPLAY_COLS = [
    ("modelid",          "Model"),
    ("dtype",            "Dtype"),
    ("Parallel",         "Parallel"),
    ("input/output",     "In/Out"),
    ("num_prompts",      "Prompts"),
    ("request_rate",     "Req/s"),
    ("token_throughput", "Throughput"),
    ("TTFT",             "TTFT(s)"),
    ("TPOT",             "TPOT(s)"),
    ("P99_TTFT",         "P99_TTFT"),
    ("P99_TPOT",         "P99_TPOT"),
]

def esc(v):
    if v is None:
        return ""
    s = str(v).strip()
    # Truncate and sanitize for markdown table cells
    s = s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    if len(s) > 80:
        s = s[:77] + "..."
    return s

with open(path, newline="") as f:
    reader = csv.reader(f, delimiter=";", quotechar='"')
    try:
        header = next(reader)
    except StopIteration:
        print("*Empty summary log.*")
        sys.exit(0)

    header = [h.strip() for h in header]
    idx = {name: header.index(name) for name in [c[0] for c in DISPLAY_COLS] if name in header}

    print("### Performance Summary")
    print("")
    print("| " + " | ".join(label for _, label in DISPLAY_COLS) + " | Status |")
    print("|" + " --- |" * (len(DISPLAY_COLS) + 1))

    total = ok = failed = 0
    for row in reader:
        if not row or not any(c.strip() for c in row):
            continue
        total += 1
        cells = []
        for key, _ in DISPLAY_COLS:
            i = idx.get(key)
            cells.append(esc(row[i]) if i is not None and i < len(row) else "")

        # Detect server-failure rows: short row, or any cell contains "Server failed"
        row_text = " ".join(row)
        if "Server failed" in row_text or "server_error" in row_text:
            status = "FAIL (server)"
            failed += 1
        elif len(row) < len(header) - 2:
            status = "FAIL"
            failed += 1
        else:
            # check if throughput cell is numeric -> success
            tp_i = idx.get("token_throughput")
            tp_val = row[tp_i].strip() if tp_i is not None and tp_i < len(row) else ""
            try:
                float(tp_val)
                status = "PASS"
                ok += 1
            except (ValueError, TypeError):
                status = "FAIL"
                failed += 1

        print("| " + " | ".join(cells) + f" | {status} |")

    print("")
    print(f"**Total:** {total} &nbsp;&nbsp; **Pass:** {ok} &nbsp;&nbsp; **Fail:** {failed}")
    print("")
