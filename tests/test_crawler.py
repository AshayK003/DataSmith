"""Tests for Schema Crawler — network-mocked, file-backed."""

import requests
from unittest.mock import MagicMock, patch


class TestDownloadFileNetworkError:
    """_download_file returns None on connection error."""

    @patch("datasmith.schema.crawler.requests.get",
           side_effect=requests.ConnectionError("DNS failure"))
    def test_download_file_network_error(self, mock_get, tmp_path):
        from datasmith.schema.crawler import _download_file

        dest = tmp_path / "test.csv"
        result = _download_file("http://example.com/test.csv", str(dest))
        assert result is None


class TestExtractSchemaFrictionlessFailure:
    """extract_schema returns empty list when frictionless.describe fails."""

    @patch("frictionless.describe",
           side_effect=ValueError("malformed CSV"))
    def test_extract_schema_frictionless_failure(self, mock_describe, tmp_path):
        from datasmith.schema.crawler import extract_schema

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")
        result = extract_schema(str(csv_path), 1)
        assert result == []


class TestReadCsvSafeTabDelimited:
    """_read_csv_safe handles tab-delimited files."""

    def test_read_csv_safe_tab_delimited(self, tmp_path):
        from datasmith.schema.crawler import _read_csv_safe

        tsv_path = tmp_path / "test.tsv"
        tsv_path.write_text("name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n")
        df = _read_csv_safe(str(tsv_path))
        assert list(df.columns) == ["name", "age", "city"]
        assert len(df) == 2


class TestFindCsvFilesDirectory:
    """_find_csv_files recursively finds only .csv files."""

    def test_find_csv_files_directory(self, tmp_path):
        from datasmith.schema.crawler import _find_csv_files

        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.csv").write_text("")
        (tmp_path / "b.txt").write_text("")
        (sub / "c.csv").write_text("")
        (sub / "d.txt").write_text("")

        result = _find_csv_files(str(tmp_path))
        assert len(result) == 2
        assert all(f.endswith(".csv") for f in result)
        assert str(tmp_path / "a.csv") in result
        assert str(sub / "c.csv") in result


class TestSeedKnowledgeGraphEmpty:
    """seed_knowledge_graph handles empty datasets dict."""

    def test_seed_knowledge_graph_empty(self):
        from datasmith.schema.crawler import seed_knowledge_graph

        mock_kg = MagicMock()
        mock_kg.upsert_domain.return_value = 1
        result = seed_knowledge_graph(mock_kg, datasets={})
        assert result == {}
