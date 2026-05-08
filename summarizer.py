import os
from pathlib import Path
from pydantic import ValidationError
from llama_cpp import Llama
from json_repair import repair_json
from models import DirectoryReport, LLMSummary

# Path to the GGUF model — sits alongside this file
_MODEL_PATH = str(Path(__file__).parent / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Load model once at import time
print(f"Loading local model: {_MODEL_PATH} ...")
llm = Llama(
    model_path=_MODEL_PATH,
    n_ctx=2048,  # TinyLlama was trained on 2048 tokens
    n_threads=os.cpu_count(),
    verbose=False,
)


def _build_report_text(report: DirectoryReport) -> str:
    """Format the DirectoryReport into a compact text block for the LLM."""
    lines = [
        "Downloads Folder Report",
        "=======================",
        f"Path: {report.path}",
        f"Scanned at: {report.scanned_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total files: {report.total_files}",
        f"Total size: {report.total_size_kb} KB ({report.total_size_kb / 1024:.2f} MB)",
        "",
        "File type breakdown:",
    ]

    for ext, count in sorted(report.breakdown_by_type.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {ext}: {count} file(s)")

    # Limit to top 10 files to stay within TinyLlama's 2048-token context
    lines.append("")
    lines.append("Top 10 files by size:")
    top_files = sorted(report.files, key=lambda f: f.size_kb, reverse=True)[:10]
    for f in top_files:
        lines.append(
            f"  - {f.name}  |  {f.size_kb} KB  |  modified {f.modified_at.strftime('%Y-%m-%d')}"
        )

    return "\n".join(lines)


_EXAMPLE_OUTPUT = """{
  "overview": "<2-3 sentence summary of the folder>",
  "notable_observations": ["<observation 1>", "<observation 2>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>"],
  "largest_file": "<filename> (<size>)",
  "most_common_type": "<ext> with <N> files"
}"""


def summarize_report(report: DirectoryReport) -> LLMSummary:
    """Run the local GGUF model and return a validated LLMSummary."""
    report_text = _build_report_text(report)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a file system analyst. "
                    "Respond with ONLY a single JSON object using exactly these 5 keys — no extra keys, no extra text:\n\n"
                    f"{_EXAMPLE_OUTPUT}"
                ),
            },
            {
                "role": "user",
                "content": f"{report_text}\n\nReturn the JSON object now:",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=900,
    )

    raw_json = response["choices"][0]["message"]["content"]
    repaired_json = repair_json(raw_json, return_objects=False)

    try:
        summary = LLMSummary.model_validate_json(repaired_json)
    except ValidationError as e:
        raise ValueError(
            f"Model returned unrecoverable JSON.\nRaw output:\n{raw_json}\n\nValidation errors:\n{e}"
        ) from e

    # Fill in fields the LLM couldn't reliably produce from real computed data
    if summary.largest_file == "N/A" and report.files:
        top = max(report.files, key=lambda f: f.size_kb)
        summary.largest_file = f"{top.name} ({top.size_kb} KB)"

    if summary.most_common_type == "N/A" and report.breakdown_by_type:
        ext, count = max(report.breakdown_by_type.items(), key=lambda x: x[1])
        summary.most_common_type = f"{ext} with {count} file(s)"

    return summary
