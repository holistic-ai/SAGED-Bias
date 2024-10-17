from functools import wraps
import os
import warnings
import pandas as pd

def clean_list(response):
    # Extract the part between the square brackets
    response_list = response[response.find('['):response.rfind(']') + 1]

    # Convert the string representation of the list to an actual list
    response_list = eval(response_list)
    return response_list

def clean_sentences_and_join(sentence_list):
    return ' '.join(sentence_list) \
        .replace('?', '').replace('.', '') \
        .replace(',', '').replace('!', '') \
        .replace(':', '').replace(';', '') \
        .replace('(', '').replace(')', '') \
        .replace('[', '').replace(']', '') \
        .replace('{', '').replace('}', '') \
        .replace('"', '').replace("'", '') \
        .replace('`', '').replace('~', '') \
        .replace('@', '').replace('#', '') \
        .replace('$', '').replace('%', '') \
        .replace('^', '').replace('&', '') \
        .replace('*', '')

def construct_non_containing_set(strings):
    def update_string_set(string_set, new_string):
        # Convert the new string to lowercase for comparison
        new_string_lower = new_string.lower()

        # Create a list of strings to remove
        strings_to_remove = [existing_string for existing_string in string_set if
                             new_string_lower in existing_string.lower()]

        # Remove all strings that contain the new string
        for string_to_remove in strings_to_remove:
            string_set.remove(string_to_remove)

        # Check if the new string should be added
        should_add = True
        for existing_string in string_set:
            if existing_string.lower() in new_string_lower:
                should_add = False
                break

        if should_add:
            string_set.add(new_string)

    result_set = set()
    for string in strings:
        update_string_set(result_set, string)
    return result_set


def check_generation_function(generation_function, test_mode=None):
    assert callable(generation_function), "The generation function must be a function."

    try:
        test_response = generation_function('test')
        assert isinstance(test_response, str), "The generation function must return a string as output."
    except TypeError as e:
        raise AssertionError("The generation function must take only one string as a positional argument.")

    if test_mode == 'list':
        try:
            test_response = generation_function('Give me a list of birds in Python format.')
            clean_list(test_response)
        except Exception as e:
            warnings.warn("The generation function seems not capable enough to respond in Python list format.")


def ignore_future_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Disable FutureWarnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            # Execute the function
            result = func(*args, **kwargs)
        finally:
            # Re-enable FutureWarnings
            warnings.filterwarnings("default", category=FutureWarning)
        return result
    return wrapper

def check_benchmark(df):
    # Assert that the DataFrame contains the required columns
    assert isinstance(df, pd.DataFrame), "Benchmark should be a DataFrame"
    assert 'keyword' in df.columns, "Benchmark must contain 'keyword' column"
    assert 'category' in df.columns, "Benchmark must contain 'category' column"
    assert 'domain' in df.columns, "Benchmark must contain 'domain' column"
    assert 'prompts' in df.columns, "Benchmark must contain 'prompts' column"
    assert 'baseline' in df.columns, "Benchmark must contain 'baseline' column"


def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the specified file path exists.
    If it does not exist, create it.

    :param file_path: The path of the file whose directory to check/create.
    """
    directory_path = os.path.dirname(file_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def _update_configuration(scheme_dict, default_dict, updated_dict):
    """
    Update the scheme dictionary with values from the updated dictionary, or from the default
    dictionary if the updated value is not available.

    Args:
    - scheme_dict (dict): The scheme dictionary with keys and None values.
    - default_dict (dict): The dictionary containing default values.
    - updated_dict (dict): The dictionary containing updated values.

    Returns:
    - dict: The configuration dictionary with updated values.
    """

    for key, value in scheme_dict.items():
        if value is None:
            if key in updated_dict:
                if isinstance(updated_dict[key], dict) and isinstance(scheme_dict[key], dict):
                    # Recursively update nested dictionaries
                    scheme_dict[key] = _update_configuration(
                        scheme_dict[key],
                        default_dict.get(key, {}),
                        updated_dict[key]
                    )
                else:
                    # Use the value from updated_dict if available
                    scheme_dict[key] = updated_dict[key]
            else:
                # Use the value from default_dict if available
                scheme_dict[key] = default_dict.get(key, None)
        elif isinstance(value, dict):
            # If the value itself is a dictionary, recursively update it
            scheme_dict[key] = _update_configuration(
                scheme_dict[key],
                default_dict.get(key, {}),
                updated_dict.get(key, {})
            )

    return scheme_dict