#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt


def parse_log_file(path):
    """
    Parse a log file like the example you gave.
    Returns:
        batches: list[int]
        cers: list[float]
        meta: dict with keys run_id, changed_param, value, filename
    """
    batches = []
    cers = []

    run_id = None
    changed_param = None
    value = None

    in_json_block = False
    json_lines = []

    with open(path, "r") as f:
        for line in f:
            line_stripped = line.strip()

            # Parse RUN line, e.g. "RUN 12: nLayers = 3"
            m_run = re.search(r"RUN\s+(\d+):\s*(.*)", line_stripped)
            if m_run:
                run_id = m_run.group(1)
                # Optional: you could also parse the "nLayers = 3" part if needed
                continue

            # Detect JSON block for changed_param/value
            if line_stripped.startswith("{"):
                in_json_block = True
                json_lines = [line_stripped]
                continue
            if in_json_block:
                json_lines.append(line_stripped)
                if line_stripped.startswith("}"):
                    in_json_block = False
                    try:
                        json_str = "\n".join(json_lines)
                        cfg = json.loads(json_str)
                        changed_param = cfg.get("changed_param", None)
                        value = cfg.get("value", None)
                    except Exception:
                        # If parsing fails, just ignore and move on
                        pass
                continue

            # Parse batch / cer lines
            # Example:
            # batch 100, ctc loss: 3.008710, cer: 0.754364, time/batch:   0.052
            m_batch = re.search(
                r"batch\s+(\d+),.*cer:\s+([0-9.+\-eE]+)", line_stripped
            )
            if m_batch:
                b = int(m_batch.group(1))
                cer = float(m_batch.group(2))
                batches.append(b)
                cers.append(cer)

    meta = {
        "run_id": run_id,
        "changed_param": changed_param,
        "value": value,
        "filename": os.path.basename(path),
    }
    return batches, cers, meta


def make_safe_tag(s):
    """Make a string safe to use in filenames."""
    if s is None:
        return "NA"
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s


# def plot_cer_curve(batches, cers, meta, out_dir):
#     """
#     Generate a single CER vs. batch plot and save as PDF (NeurIPS-friendly).
#     """
#     if not batches:
#         print(f"[WARN] No batch lines found for {meta['filename']}, skipping.")
#         return

#     # NeurIPS single-column width is ~3.25in; height can be ~2.2–2.5in
#     fig, ax = plt.subplots(figsize=(3.25, 2.25))

#     ax.plot(batches, cers)

#     ax.set_xlabel("Training batch")
#     ax.set_ylabel("CER (validation)")

#     # Build a concise title
#     title_parts = []
#     if meta["changed_param"] is not None and meta["value"] is not None:
#         title_parts.append(f"{meta['changed_param']} = {meta['value']}")
#     if meta["run_id"] is not None:
#         title_parts.append(f"Run {meta['run_id']}")
#     if title_parts:
#         ax.set_title(", ".join(title_parts), fontsize=9)

#     # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

#     fig.tight_layout()

#     run_tag = make_safe_tag(meta["run_id"])
#     param_tag = make_safe_tag(meta["changed_param"])
#     value_tag = make_safe_tag(meta["value"])
#     base_name = f"cer_curve_run{run_tag}_{param_tag}_{value_tag}.pdf"

#     out_path = os.path.join(out_dir, base_name)
#     fig.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)

#     print(f"[INFO] Saved CER curve to {out_path}")

def plot_cer_curve(batches, cers, meta, out_dir):
    """
    Generate a single CER vs. batch plot and save as PDF (NeurIPS-friendly).
    Also annotates the CER at the final batch (e.g., batch=9900).
    """
    if not batches:
        print(f"[WARN] No batch lines found for {meta['filename']}, skipping.")
        return

    # NeurIPS single-column width is ~3.25in
    fig, ax = plt.subplots(figsize=(3.25, 2.25))

    ax.plot(batches, cers, lw=1.5)

    ax.set_xlabel("Training batch")
    ax.set_ylabel("CER (validation)")

    # Title (compact NeurIPS style)
    title_parts = []
    if meta["changed_param"] and meta["value"]:
        title_parts.append(f"{meta['changed_param']} = {meta['value']}")
    if meta["run_id"]:
        title_parts.append(f"Run {meta['run_id']}")
    ax.set_title(", ".join(title_parts), fontsize=9)

    # -------------------------------
    # Highlight final CER value
    # -------------------------------
    final_batch = batches[-1]
    final_cer = cers[-1]

    ax.scatter([final_batch], [final_cer], color="red", s=5, zorder=5)
    ax.text(
        final_batch,
        final_cer + 0.01,               # slightly above the point
        f"CER={final_cer:.4f}",
        fontsize=8,
        ha="right"
    )

    # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    # filename-safe tags
    run_tag = make_safe_tag(meta["run_id"])
    param_tag = make_safe_tag(meta["changed_param"])
    value_tag = make_safe_tag(meta["value"])

    base_name = f"cer_curve_run{run_tag}_{param_tag}_{value_tag}.pdf"
    out_path = os.path.join(out_dir, base_name)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved CER curve to {out_path} (final CER={final_cer:.4f})")

def main():
    parser = argparse.ArgumentParser(
        description="Plot CER vs. training batches for each log file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing .txt log files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save output figures (PDF).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.txt")))
    if not log_files:
        print(f"[WARN] No .txt files found in {args.log_dir}")
        return

    for path in log_files:
        print(f"[INFO] Processing {path}")
        batches, cers, meta = parse_log_file(path)
        plot_cer_curve(batches, cers, meta, args.out_dir)


if __name__ == "__main__":
    main()
