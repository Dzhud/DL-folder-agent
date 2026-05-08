import os
from pathlib import Path

from scanner import scan_directory
from summarizer import summarize_report


def print_report(report, summary) -> None:
    print("\n" + "=" * 55)
    print("  DOWNLOADS FOLDER REPORT")
    print("=" * 55)
    print(f"  Path        : {report.path}")
    print(f"  Scanned at  : {report.scanned_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total files : {report.total_files}")
    print(f"  Total size  : {report.total_size_kb} KB ({report.total_size_kb / 1024:.2f} MB)")

    print("\n--- File Type Breakdown ---")
    for ext, count in sorted(report.breakdown_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext:<20} {count} file(s)")

    print("\n--- LLM Summary ---")
    print(f"\nOverview:\n  {summary.overview}")

    print("\nNotable Observations:")
    for obs in summary.notable_observations:
        print(f"  • {obs}")

    print("\nRecommendations:")
    for rec in summary.recommendations:
        print(f"  • {rec}")

    print(f"\nLargest File  : {summary.largest_file}")
    print(f"Most Common Type: {summary.most_common_type}")
    print("=" * 55 + "\n")


def main():
    downloads_path = str(Path.home() / "Downloads")

    if not os.path.isdir(downloads_path):
        print(f"Directory not found: {downloads_path}")
        return

    print(f"Scanning: {downloads_path} ...")
    report = scan_directory(downloads_path)

    print(f"Found {report.total_files} files. Sending to LLM for summary...")
    summary = summarize_report(report)

    print_report(report, summary)


if __name__ == "__main__":
    main()
