from pydantic import BaseModel, Field, field_validator
from datetime import datetime


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
