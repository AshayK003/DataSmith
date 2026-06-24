"""Pydantic models for the Schema Knowledge Graph."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Domain(BaseModel):
    id: Optional[int] = None
    name: str
    description: str = ""
    parent_domain_id: Optional[int] = None
    created_at: Optional[datetime] = None


class DatasetSchema(BaseModel):
    id: Optional[int] = None
    source: str = "kaggle"
    source_url: str = ""
    domain_id: Optional[int] = None
    dataset_name: str = ""
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    crawled_at: Optional[datetime] = None


class ColumnSchema(BaseModel):
    id: Optional[int] = None
    dataset_id: int
    column_name: str
    column_description: str = ""
    data_type: str = ""           # numeric, categorical, text, datetime, boolean
    null_ratio: Optional[float] = None
    distinct_count: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    sample_values: str = "[]"     # JSON array


class DomainProfile(BaseModel):
    id: Optional[int] = None
    domain_id: int
    profile_json: str = "{}"
