"""Tests for the Schema Knowledge Graph."""

import json
import tempfile

import pytest

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.schema.models import ColumnSchema, DatasetSchema


@pytest.fixture
def db():
    """Fresh temp database for each test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    database = Database(tmp.name)
    yield database
    database.close()


@pytest.fixture
def kg(db):
    return KnowledgeGraph(db)


class TestDomains:
    def test_upsert_creates_domain(self, kg):
        did = kg.upsert_domain("e-commerce", "Retail data")
        domain = kg.get_domain(did)
        assert domain.name == "e-commerce"
        assert domain.description == "Retail data"

    def test_upsert_returns_existing_id(self, kg):
        d1 = kg.upsert_domain("finance")
        d2 = kg.upsert_domain("finance")
        assert d1 == d2

    def test_get_domain_by_name(self, kg):
        kg.upsert_domain("healthcare", "Medical data")
        domain = kg.get_domain_by_name("healthcare")
        assert domain is not None
        assert domain.description == "Medical data"

    def test_get_domain_by_name_missing(self, kg):
        assert kg.get_domain_by_name("nope") is None

    def test_list_domains(self, kg):
        kg.upsert_domain("a")
        kg.upsert_domain("b")
        names = [d.name for d in kg.list_domains()]
        assert names == ["a", "b"]


class TestDatasets:
    def test_upsert_dataset(self, kg):
        did = kg.upsert_domain("e-commerce")
        ds = DatasetSchema(source="kaggle", source_url="kaggle.com/ds1",
                           domain_id=did, dataset_name="Test DS")
        dsid = kg.upsert_dataset(ds)
        loaded = kg.get_dataset(dsid)
        assert loaded.dataset_name == "Test DS"
        assert loaded.source == "kaggle"

    def test_upsert_returns_existing(self, kg):
        did = kg.upsert_domain("e-commerce")
        ds = DatasetSchema(source="kaggle", source_url="kaggle.com/ds1",
                           domain_id=did, dataset_name="Test")
        d1 = kg.upsert_dataset(ds)
        d2 = kg.upsert_dataset(ds)
        assert d1 == d2

    def test_list_datasets_by_domain(self, kg):
        d1 = kg.upsert_domain("finance")
        d2 = kg.upsert_domain("health")
        for i in range(3):
            kg.upsert_dataset(DatasetSchema(
                source="kaggle", source_url=f"url_{i}", domain_id=d1,
                dataset_name=f"DS{i}"))
        kg.upsert_dataset(DatasetSchema(
            source="kaggle", source_url="url_h", domain_id=d2,
            dataset_name="Health"))
        fin_ds = kg.list_datasets(domain_id=d1)
        assert len(fin_ds) == 3
        all_ds = kg.list_datasets()
        assert len(all_ds) == 4


class TestColumns:
    def test_insert_and_read_columns(self, kg):
        did = kg.upsert_domain("e-commerce")
        dsid = kg.upsert_dataset(DatasetSchema(
            source="kaggle", source_url="kaggle.com/test",
            domain_id=did, dataset_name="Test"))
        cols = [
            ColumnSchema(dataset_id=dsid, column_name="price",
                         data_type="numeric", null_ratio=0.05,
                         mean=100.0, minimum=0.0, maximum=1000.0),
            ColumnSchema(dataset_id=dsid, column_name="name",
                         data_type="text"),
        ]
        kg.insert_columns(cols)
        # Verify via domain schema query (production code path)
        schema = kg.get_column_schemas_for_domain("e-commerce")
        assert schema is not None
        names = {c["column_name"] for c in schema}
        assert "price" in names
        assert "name" in names


class TestLLMCache:
    def test_cache_miss(self, kg):
        assert kg.llm_cache_get("nonexistent") is None

    def test_cache_hit(self, kg):
        key = "e-commerce transactions schema"
        response = {"domain": "e-commerce", "columns": [{"name": "price"}]}
        kg.llm_cache_set(key, response, model="gpt-4o-mini")
        cached = kg.llm_cache_get(key)
        assert cached == response

    def test_cache_different_keys(self, kg):
        kg.llm_cache_set("key1", {"a": 1})
        kg.llm_cache_set("key2", {"b": 2})
        assert kg.llm_cache_get("key1") == {"a": 1}
        assert kg.llm_cache_get("key2") == {"b": 2}


class TestDomainProfiles:
    def test_upsert_profile(self, kg):
        did = kg.upsert_domain("iot-sensors")
        profile = json.dumps({"null_correlations": {"temp": 0.1}})
        pid = kg.upsert_domain_profile(did, profile)
        loaded = kg.get_domain_profile(did)
        assert loaded is not None
        assert "null_correlations" in loaded.profile_json

    def test_profile_update(self, kg):
        did = kg.upsert_domain("finance")
        kg.upsert_domain_profile(did, json.dumps({"v1": 1}))
        kg.upsert_domain_profile(did, json.dumps({"v2": 2}))
        loaded = kg.get_domain_profile(did)
        data = json.loads(loaded.profile_json)
        assert data == {"v2": 2}


class TestStats:
    def test_empty_stats(self, kg):
        stats = kg.stats()
        assert stats["domains"] == 0
        assert stats["datasets"] == 0
        assert stats["columns"] == 0

    def test_stats_after_insert(self, kg):
        did = kg.upsert_domain("e-commerce")
        dsid = kg.upsert_dataset(DatasetSchema(
            source="kaggle", source_url="url", domain_id=did,
            dataset_name="Test"))
        kg.insert_columns([
            ColumnSchema(dataset_id=dsid, column_name="c1"),
            ColumnSchema(dataset_id=dsid, column_name="c2"),
        ])
        stats = kg.stats()
        assert stats["domains"] == 1
        assert stats["datasets"] == 1
        assert stats["columns"] == 2
