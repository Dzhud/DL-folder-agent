import json

from openai import OpenAI

from models import DirectoryReport, LLMSummary

client = OpenAI()  # reads OPENAI_API_KEY from environment


def _build_report_text(report: DirectoryReport) -> str:
    """Format the DirectoryReport into a readable text block for the LLM."""
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

    lines.append("")
    lines.append("Top 20 files by size:")
    top_files = sorted(report.files, key=lambda f: f.size_kb, reverse=True)[:20]
    for f in top_files:
        lines.append(
            f"  - {f.name}  |  {f.size_kb} KB  |  modified {f.modified_at.strftime('%Y-%m-%d')}"
        )

    return "\n".join(lines)


def summarize_report(report: DirectoryReport) -> LLMSummary:
    """Send the report to OpenAI and return a validated LLMSummary."""
    report_text = _build_report_text(report)
    schema = json.dumps(LLMSummary.model_json_schema(), indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that analyzes file system reports. "
                    "Return your analysis as a JSON object exactly matching this schema:\n\n"
                    f"{schema}"
                ),
            },
            {
                "role": "user",
                "content": f"Analyze this Downloads folder report:\n\n{report_text}",
            },
        ],
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content
    return LLMSummary.model_validate_json(raw_json)
