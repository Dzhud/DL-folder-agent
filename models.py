from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from typing import Any


class FileInfo(BaseModel):
    name: str
    extension: str
    size_kb: float
    modified_at: datetime

    @field_validator("extension", mode="before")
    @classmethod
    def normalize_extension(cls, v: str) -> str:
        return v.lower() if v else "no_extension"


class DirectoryReport(BaseModel):
    path: str
    scanned_at: datetime = Field(default_factory=datetime.now)
    total_files: int
    total_size_kb: float
    files: list[FileInfo]
    breakdown_by_type: dict[str, int]  # e.g. {".pdf": 5, ".mp4": 3}


class LLMSummary(BaseModel):
    overview: str = Field(description="A 2-3 sentence overview of the Downloads folder contents")
    notable_observations: list[str] = Field(description="Interesting patterns or standout facts about the files")
    recommendations: list[str] = Field(description="Actionable suggestions, e.g. files to delete or organize")
    largest_file: str = Field(description="Name and size of the largest file")
    most_common_type: str = Field(description="The most frequently occurring file type and its count")

    @model_validator(mode="before")
    @classmethod
    def _fix_field_aliases(cls, data: Any) -> Any:
        """Tolerate common field-name variations from small LLMs."""
        if isinstance(data, dict):
            if "large_file" in data and "largest_file" not in data:
                data["largest_file"] = data.pop("large_file")
            if "common_type" in data and "most_common_type" not in data:
                data["most_common_type"] = data.pop("common_type")
        return data
