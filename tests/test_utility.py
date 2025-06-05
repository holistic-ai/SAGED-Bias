import warnings

import pytest
import os
import pandas as pd
from saged import (
    clean_list,
    clean_sentences_and_join,
    construct_non_containing_set,
    check_generation_function,
    ignore_future_warnings,
    check_benchmark,
    ensure_directory_exists,
    _update_configuration,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "keyword": ["test_keyword"],
        "concept": ["test_concept"],
        "domain": ["test_domain"],
        "prompts": ["test_prompt"],
        "baseline": ["test_baseline"]
    })


def test_clean_list():
    response = "Here is a list: [1, 2, 3, 4]"
    result = clean_list(response)
    assert result == [1, 2, 3, 4], "clean_list did not extract the list correctly."


def test_clean_sentences_and_join():
    sentences = ["Hello,", "world!", "This", "is", "a", "test."]
    result = clean_sentences_and_join(sentences)
    assert result == "Hello world This is a test", "clean_sentences_and_join did not clean and join correctly."


# def test_construct_non_containing_set():
#     strings = ["cat", "dog", "caterpillar", "bird"]
#     result = construct_non_containing_set(strings)
#     assert result == {"dog", "caterpillar", "bird"}, "construct_non_containing_set did not filter correctly."
#

def test_check_generation_function():
    def mock_generation_function(prompt):
        if "list" in prompt:
            return "[1, 2, 3, 4]"
        return "test response"

    # Check valid function
    check_generation_function(mock_generation_function)

    # Check list capability
    check_generation_function(mock_generation_function, test_mode="list")

    # Invalid function
    with pytest.raises(AssertionError):
        check_generation_function("not a function")


def test_ignore_future_warnings():
    @ignore_future_warnings
    def function_with_warning():
        warnings.warn("This is a FutureWarning", category=FutureWarning)

    # No warnings should be raised
    function_with_warning()


def test_check_benchmark(sample_dataframe):
    # Valid case
    check_benchmark(sample_dataframe)

    # Invalid DataFrame
    with pytest.raises(AssertionError):
        check_benchmark(pd.DataFrame({"invalid_column": [1]}))


def test_ensure_directory_exists(tmpdir):
    file_path = os.path.join(tmpdir, "test_dir", "test_file.txt")

    # Ensure directory does not exist initially
    assert not os.path.exists(os.path.dirname(file_path))

    # Create the directory
    ensure_directory_exists(file_path)

    # Ensure directory now exists
    assert os.path.exists(os.path.dirname(file_path))


def test_update_configuration():
    scheme_dict = {"key1": None, "key2": {"subkey1": None}}
    default_dict = {"key1": "default_value1", "key2": {"subkey1": "default_subvalue1"}}
    updated_dict = {"key2": {"subkey1": "updated_subvalue1"}}

    result = _update_configuration(scheme_dict, default_dict, updated_dict)

    assert result["key1"] == "default_value1", "_update_configuration did not update key1 correctly."
    assert result["key2"]["subkey1"] == "updated_subvalue1", "_update_configuration did not update key2.subkey1 correctly."