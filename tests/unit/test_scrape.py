import pytest
from unittest.mock import patch, MagicMock
from saged import (
    find_similar_keywords,
    search_wikipedia,
    KeywordFinder,
    SourceFinder,
    Scraper,
    SAGEDData
)
from sentence_transformers import SentenceTransformer


@pytest.fixture
def valid_saged_data_for_scraper():
    data = SAGEDData.create_data(
        domain="test_domain",
        concept="test_concept",
        data_tier="scraped_sentences",
        data=[
            {
                "concept": "test_concept",
                "domain": "test_domain",
                "concept_shared_source": [
                    {
                        "source_tag": "default",
                        "source_type": "wiki_urls",
                        "source_specification": ["http://mock-url.com"],
                    }
                ],
                "keywords": {
                    "test_keyword": {
                        "scraped_sentences": [],
                        "keyword_type": "sub-concepts",
                        "keyword_provider": "manual",
                        "scrap_mode": "in_page",
                        "scrap_shared_area": "Yes",
                    }
                },
            }
        ],
    )
    return data


@patch("sentence_transformers.SentenceTransformer")
def test_find_similar_keywords(mock_model):
    mock_instance = MagicMock()
    mock_instance.encode.side_effect = lambda x: [0.5] * len(x) if isinstance(x, list) else [0.8]
    mock_model.return_value = mock_instance

    keywords_list = ["apple", "banana", "cherry"]
    target_word = "apple"

    result = find_similar_keywords("paraphrase-MiniLM-L6-v2", target_word, keywords_list, top_n=2)
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


# @patch("requests.get")
# @patch("bs4.BeautifulSoup")
# def test_scrape_in_page_for_wiki_with_buffer_files(
#     mock_soup, mock_requests, valid_saged_data_for_scraper
# ):
#     scraper = Scraper(valid_saged_data_for_scraper)
#     mock_requests.return_value.content = "<html><body><p>Test keyword content</p></body></html>"
#     mock_soup.return_value.find_all.return_value = [
#         MagicMock(get_text=lambda: "Test keyword content")
#     ]
#
#     scraper.scrape_in_page_for_wiki_with_buffer_files()
#     assert len(scraper.data[0]["keywords"]["test_keyword"]["scraped_sentences"]) > 0, \
#         "No sentences were scraped"


# @patch("glob.glob")
# def test_find_scrape_paths_local(mock_glob, valid_saged_data_for_scraper):
#     source_finder = SourceFinder(valid_saged_data_for_scraper)
#     mock_glob.return_value = ["/mock/path/file1.txt", "/mock/path/file2.txt"]
#
#     saged_data = source_finder.find_scrape_paths_local("/mock/path")
#     assert len(saged_data.data[0]["category_shared_source"][0]["source_specification"]) == 2, \
#         "Incorrect number of local paths found"
#     assert saged_data.data[0]["category_shared_source"][0]["source_type"] == "local_paths", \
#         "Incorrect source type for local paths"