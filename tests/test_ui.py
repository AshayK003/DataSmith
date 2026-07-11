from unittest.mock import patch

from datasmith.ui.icons import (
    _svg,
    BRAND,
    INFO,
    DATABASE,
    WAVES,
    DOWNLOAD,
    SPARKLES,
)
from datasmith.ui.components import render_header


class TestSvg:
    def test_default_size(self):
        result = _svg('<path d="M0 0"/>')
        assert 'width="18"' in result
        assert 'height="18"' in result

    def test_custom_size(self):
        result = _svg('<path d="M0 0"/>', size=32)
        assert 'width="32"' in result
        assert 'height="32"' in result

    def test_valid_svg_structure(self):
        result = _svg('<path d="M0 0"/>')
        assert result.startswith("<svg")
        assert result.strip().endswith("</svg>")
        assert 'xmlns="http://www.w3.org/2000/svg"' in result

    def test_contains_path_data(self):
        path = '<circle cx="12" cy="12" r="10"/>'
        result = _svg(path)
        assert path in result

    def test_required_attributes(self):
        result = _svg('<path d="M0 0"/>')
        assert 'viewBox="0 0 24 24"' in result
        assert 'fill="none"' in result
        assert 'stroke="currentColor"' in result

    def test_module_constants_are_svgs(self):
        for icon in [BRAND, INFO, DATABASE, WAVES, DOWNLOAD, SPARKLES]:
            assert isinstance(icon, str)
            assert icon.startswith("<svg")


class TestRenderHeader:
    @patch("datasmith.ui.components.st")
    def test_default_active_is_home(self, mock_st):
        render_header()

        html = mock_st.markdown.call_args[0][0]

        assert "Home" in html
        assert "ds-nav-btn active" in html

    @patch("datasmith.ui.components.st")
    def test_generate_active(self, mock_st):
        render_header("generate")

        html = mock_st.markdown.call_args[0][0]

        assert "Generate" in html
        assert "ds-nav-btn active" in html

    @patch("datasmith.ui.components.st")
    def test_about_active(self, mock_st):
        render_header("about")

        html = mock_st.markdown.call_args[0][0]

        assert "About" in html
        assert "ds-nav-btn active" in html
        