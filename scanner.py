from pathlib import Path
from datetime import datetime

from models import FileInfo, DirectoryReport


def scan_directory(path: str) -> DirectoryReport:
    """Scan a directory and return a structured DirectoryReport."""
    dir_path = Path(path)
    files: list[FileInfo] = []
    breakdown: dict[str, int] = {}

    for entry in dir_path.iterdir():
        if entry.is_file():
            stat = entry.stat()
            ext = entry.suffix.lower() or "no_extension"
            files.append(
                FileInfo(
                    name=entry.name,
                    extension=ext,
                    size_kb=round(stat.st_size / 1024, 2),
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                )
            )
            breakdown[ext] = breakdown.get(ext, 0) + 1

    total_size = round(sum(f.size_kb for f in files), 2)

    return DirectoryReport(
        path=str(dir_path.resolve()),
        total_files=len(files),
        total_size_kb=total_size,
        files=files,
        breakdown_by_type=breakdown,
    )
