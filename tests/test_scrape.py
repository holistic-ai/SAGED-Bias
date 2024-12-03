import pytest
from unittest.mock import patch, MagicMock
from saged import (
    find_similar_keywords,
    search_wikipedia,
    KeywordFinder,
    SourceFinder,
    Scraper
)
from sentence_transformers import SentenceTransformer
import os


@pytest.fixture
def sample_keyword_data():
    return {
        "category": "test_category",
        "domain": "test_domain",
        "data_tier": "keywords",
        "data": [{"category": "test_category", "domain": "test_domain", "keywords": {"test": {}}}]
    }


@patch("sentence_transformers.SentenceTransformer")
def test_find_similar_keywords(mock_model):
    mock_instance = MagicMock()
    mock_instance.encode.side_effect = lambda x: [0.5] * len(x) if isinstance(x, list) else [0.8]
    mock_model.return_value = mock_instance

    keywords_list = ["apple", "banana", "cherry"]
    target_word = "apple"

    result = find_similar_keywords("mock_model", target_word, keywords_list, top_n=2)
    assert len(result) == 2, "Top N keywords were not returned"
    assert "apple" in result, "Target word was not in the result"


@patch("wikipediaapi.Wikipedia")
def test_search_wikipedia(mock_wikipedia):
    mock_page = MagicMock()
    mock_page.exists.return_value = True
    mock_page.title = "Mock Page"
    mock_page.text = "Mock text content"
    mock_wikipedia.return_value.page.return_value = mock_page

    result, _ = search_wikipedia("Mock Topic", language="en")
    assert result.title == "Mock Page", "Wikipedia page title mismatch"


def test_keyword_finder_keywords_to_saged_data(sample_keyword_data):
    keyword_finder = KeywordFinder(category="test_category", domain="test_domain")
    keyword_finder.keywords = ["keyword1", "keyword2"]
    keyword_finder.finder_mode = "embedding"

    saged_data = keyword_finder.keywords_to_saged_data()
    assert saged_data.data[0]["keywords"]["keyword1"]["keyword_type"] == "sub-concepts", \
        "Keyword metadata is incorrect"
    assert saged_data.data[0]["keywords"]["keyword2"]["keyword_provider"] == "embedding", \
        "Keyword provider metadata is incorrect"


@patch("requests.get")
@patch("bs4.BeautifulSoup")
def test_scrape_in_page_for_wiki_with_buffer_files(mock_soup, mock_requests, sample_keyword_data, tmpdir):
    scraper = Scraper(sample_keyword_data)
    mock_requests.return_value.content = "<html><body><p>Test keyword content</p></body></html>"
    mock_soup.return_value.find_all.return_value = [
        MagicMock(get_text=lambda: "Test keyword content")
    ]

    scraper.source_finder = [{"source_type": "wiki_urls", "source_specification": ["http://mock-url.com"]}]
    scraper.keywords = ["keyword"]

    temp_dir = tmpdir.mkdir("temp_results")
    scraper.scrape_in_page_for_wiki_with_buffer_files()
    assert len(scraper.data[0]["keywords"]["keyword"]["scraped_sentences"]) > 0, \
        "No sentences were scraped"


@patch("glob.glob")
def test_find_scrape_paths_local(mock_glob, sample_keyword_data):
    source_finder = SourceFinder(sample_keyword_data)
    mock_glob.return_value = ["/mock/path/file1.txt", "/mock/path/file2.txt"]

    saged_data = source_finder.find_scrape_paths_local("/mock/path")
    assert len(saged_data.data[0]["category_shared_source"][0]["source_specification"]) == 2, \
        "Incorrect number of local paths found"
    assert saged_data.data[0]["category_shared_source"][0]["source_type"] == "local_paths", \
        "Incorrect source type for local paths"