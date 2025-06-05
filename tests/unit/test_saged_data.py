import pytest
import pandas as pd
import json
from saged import SAGEDData


@pytest.fixture
def sample_data():
    # Sample data for tests
    return [
        {
            "concept": "test_concept",
            "domain": "test_domain",
            "keywords": {
                "keyword1": {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": "manual",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes"
                }
            },
            "concept_shared_source": [
                {
                    "source_tag": "default",
                    "source_type": "unknown",
                    "source_specification": []
                }
            ]
        }
    ]

@pytest.fixture
def sample_dataframe():
    # Sample dataframe for split_sentences or questions
    return pd.DataFrame({
        "keyword": ["keyword1"],
        "concept": ["test_concept"],
        "domain": ["test_domain"],
        "prompts": ["sample prompt"],
        "baseline": ["sample baseline"]
    })

def test_init():
    instance = SAGEDData("test_domain", "test_concept", "keywords")
    assert instance.domain == "test_domain"
    assert instance.concept == "test_concept"
    assert instance.data_tier == "keywords"
    assert isinstance(instance.data, list)

def test_check_format_keywords(sample_data):
    # Validate check_format for keywords
    SAGEDData.check_format("keywords", sample_data)

def test_check_format_dataframe(sample_dataframe):
    # Validate check_format for split_sentences or questions
    SAGEDData.check_format("split_sentences", sample_dataframe)

def test_create_data_keywords():
    # Validate create_data for keywords tier
    instance = SAGEDData.create_data("test_domain", "test_concept", "keywords")
    assert instance.data[0]["keywords"] == {}

def test_create_data_split_sentences():
    # Validate create_data for split_sentences tier
    instance = SAGEDData.create_data("test_domain", "test_concept", "split_sentences")
    assert isinstance(instance.data, pd.DataFrame)
    assert list(instance.data.columns) == ["keyword", "concept", "domain", "prompts", "baseline", "source_tag"]

def test_load_file(tmpdir, sample_data):
    # Test loading a JSON file
    test_file = tmpdir.join("test_file.json")
    with open(test_file, "w") as f:
        json.dump(sample_data, f)

    instance = SAGEDData.load_file("test_domain", "test_concept", "keywords", str(test_file))
    assert instance is not None
    assert instance.data == sample_data

def test_save_file(tmpdir, sample_data):
    # Test saving a JSON file
    instance = SAGEDData("test_domain", "test_concept", "keywords")
    instance.data = sample_data

    test_file = tmpdir.join("output.json")
    instance.save(str(test_file))

    with open(test_file, "r") as f:
        saved_data = json.load(f)

    assert saved_data == sample_data

def test_show(capsys, sample_data):
    # Test show method
    instance = SAGEDData("test_domain", "test_concept", "keywords")
    instance.data = sample_data

    instance.show("short", data_tier="keywords")
    captured = capsys.readouterr()
    assert "Keywords: keyword1" in captured.out

def test_add_keyword(sample_data):
    # Test adding a keyword
    instance = SAGEDData("test_domain", "test_concept", "keywords")
    instance.data = sample_data

    instance.add(keyword="new_keyword")
    assert "new_keyword" in instance.data[0]["keywords"]

def test_remove_keyword(sample_data):
    # Test removing a keyword
    instance = SAGEDData("test_domain", "test_concept", "keywords")
    instance.data = sample_data

    instance.remove("keyword1", data_tier="keywords")
    assert "keyword1" not in instance.data[0]["keywords"]

def test_sub_sample(sample_dataframe):
    # Test sub-sampling a DataFrame
    instance = SAGEDData.create_data("test_domain", "test_concept", "split_sentences", sample_dataframe)
    sub_sampled = instance.sub_sample(sample=1, saged_format=True)
    assert len(sub_sampled.data) == 1

def test_model_generation(sample_dataframe):
    # Test model_generation method
    def mock_generation_function(prompt):
        return f"Generated: {prompt}"

    instance = SAGEDData.create_data("test_domain", "test_concept", "split_sentences", sample_dataframe)
    instance.model_generation(mock_generation_function, "generated_output")
    assert "generated_output" in instance.data.columns
    assert instance.data["generated_output"][0] == "Generated: sample prompt"