import os

import requests
from bs4 import BeautifulSoup
import glob

import wikipediaapi
import numpy as np
import warnings
from ._saged_data import SAGEDData as saged_data

from sentence_transformers import SentenceTransformer, util

import re
from tqdm import tqdm

from ._utility import clean_list, construct_non_containing_set, check_generation_function
from ._utility import ignore_future_warnings

import tempfile
import shutil

@ignore_future_warnings
def find_similar_keywords(model_name, target_word, keywords_list, top_n=100):
    """
    Find the top N keywords most similar to the target word.

    Args:
    - model_name (str): The name of the pre-trained model to use.
    - target_word (str): The word for which we want to find similar keywords.
    - keywords_list (list): The list containing the keywords.
    - n_keywords (int): The number of top similar keywords to return (default is 100).

    Returns:
    - list: The top N keywords most similar to the target word.
    """
    # Load pre-trained model
    model = SentenceTransformer(model_name)

    # Embed the keywords and the target word
    keyword_embeddings = model.encode(keywords_list)
    target_embedding = model.encode(target_word)

    # Compute cosine similarities
    cosine_similarities = util.cos_sim(target_embedding, keyword_embeddings)[0].numpy()

    # Find the top N keywords most similar to the target word
    top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    top_keywords = [keywords_list[i] for i in top_indices]

    return top_keywords

@ignore_future_warnings
def search_wikipedia(topic, language='en', user_agent='SAGED-bias (contact@holisticai.com)'):
    """
    Search for a topic on Wikipedia and return the page object.

    Args:
    - topic (str): The topic to search for.
    - language (str): Language of the Wikipedia (default is 'en').
    - user_agent (str): User agent string to use for the API requests.

    Returns:
    - Wikipedia page object or an error message if the page does not exist.
    """
    wiki_wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)
    page = wiki_wiki.page(topic)

    if not page.exists():
        return f"No Wikipedia page found for {topic}"

    return page, wiki_wiki


class KeywordFinder:
    def __init__(self, concept, domain, use_database=False, database_config=None):
        self.concept = concept
        self.domain = domain
        self.model = ''
        self.keywords = []
        self.finder_mode = None
        self.kw_targeted_source_finder_dict = None
        self.use_database = use_database
        self.database_config = database_config or {}

    def to_saged_data(self):
        keywords = self.keywords

        if self.finder_mode == "llm" or self.finder_mode == "embedding":
            keywords_dictionary = {}
            if self.finder_mode == "llm":
                keyword_provider = f'{self.finder_mode}_{self.model}'
            else:
                keyword_provider = f'{self.finder_mode}'

            for keyword in keywords:
                keywords_dictionary[keyword] = {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": keyword_provider,
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes"
                }

                if isinstance(self.kw_targeted_source_finder_dict, dict):
                    targeted_source_finder = self.kw_targeted_source_finder_dict[keyword]
                    saged_data.check_format(source_finder_only=True)(targeted_source_finder)
                    keywords_dictionary[keyword]["targeted_source_finder"] = targeted_source_finder

            keyword_entry = {
                "concept": self.concept,
                "domain": self.domain,
                "keywords": keywords_dictionary
            }

            keywords_saged_data = saged_data.create_data(concept=self.concept, domain=self.domain, data_tier='keywords',
                                                   data=[keyword_entry])
            keywords_saged_data.use_database = self.use_database
            keywords_saged_data.database_config = self.database_config

            return keywords_saged_data

    def lm_inquiries(self, n_run=20,
                     n_keywords=20,
                     generation_function=None,
                     model_name=None,
                     embedding_model=None,
                     show_progress=True):

        def get_llm_template(template_type, **kwargs):
            concept = kwargs.get('concept')
            domain = kwargs.get('domain')

            if concept and domain:
                template_people = f"""
                Given {concept} under the topic {domain}, \
                provide a list of famous names that are representative to the {concept}.
                Output your answer in a python list format only and nothing else.
                """

                template_people_short = f"""
                Given {concept} under the topic {domain}, \
                provide a list of famous names that are representative to {concept}.
                These names can be short family names for which they are identified within the context.
                Output your answer in a python list format only and nothing else.
                """

                template_characteristics = f"""
                Given the {concept} under the topic {domain}, \
                provide a list of characteristics that are necessary (!!!) to {concept}.
                Output your answer in a python list format only and nothing else.
                """

                template_subcategories = f"""
                Given the {concept} under the topic {domain}, \
                provide a list of sub-categories of {concept}.
                Output your answer in a python list format only and nothing else.
                """

                template_syn = f"""
                Given the {concept} under the topic {domain}, \
                provide a list of synonyms of {concept}.
                Output your answer in a python list format only and nothing else.
                """

                template_root = f"""
                Given the {concept} under the topic {domain}, \
                provide a list of words that share the same grammatical roots with {concept}.
                Output your answer in a python list format only and nothing else.
                """

                if template_type == 'people':
                    return template_people
                elif template_type == 'people_short':
                    return template_people_short
                elif template_type == 'characteristics':
                    return template_characteristics
                elif template_type == 'subcategories':
                    return template_subcategories
                elif template_type == 'synonym':
                    return template_syn
                elif template_type == 'root':
                    return template_root

            print('Template type not found')
            return None

        concept = self.concept
        domain = self.domain
        if model_name:
            self.model = model_name
        else:
            warnings.warn("Model name not provided. Using the default model name 'user_LLM'")
            self.model = 'user_LLM'
        final_set = {concept}
        check_generation_function(generation_function, test_mode='list')

        for _ in tqdm(range(n_run), desc='finding keywords by LLM', unit='run'):
            try:
                if _ == 0 or _ == 1:
                    response = clean_list(
                        generation_function(get_llm_template('root', concept=concept, domain=domain)))
                    final_set.update(response)
                if _ % 5 == 0:
                    # response = clean_list(agent.invoke(get_template('people_short', concept=concept, domain=domain)))
                    response = clean_list(
                        generation_function(get_llm_template('subcategories', concept=concept, domain=domain)))
                elif _ % 5 == 1:
                    # response = clean_list(agent.invoke(get_template('people', concept=concept, domain=domain)))
                    response = clean_list(
                        generation_function(get_llm_template('characteristics', concept=concept, domain=domain)))
                elif _ % 5 == 2:
                    response = clean_list(
                        generation_function(get_llm_template('synonym', concept=concept, domain=domain)))
                elif _ % 5 == 3:
                    response = clean_list(
                        generation_function(get_llm_template('people', concept=concept, domain=domain)))
                elif _ % 5 == 4:
                    response = clean_list(
                        generation_function(get_llm_template('people_short', concept=concept, domain=domain)))
                if show_progress:
                    print(f"Response: {response}")
                # Extend the final_set with the response_list
                final_set.update(response)

            except Exception as e:
                print(f"Invocation failed at iteration {_}: {e}")

        self.keywords = list(construct_non_containing_set(list(final_set)))
        if show_progress:
            print('final set of keywords:')
            print(self.keywords)
        if len(self.keywords) > n_keywords:
            if embedding_model:
                self.keywords = find_similar_keywords(embedding_model, concept, self.keywords, top_n=n_keywords)
            else:
                self.keywords = find_similar_keywords('paraphrase-MiniLM-L6-v2', self.concept, self.keywords,
                                                      top_n=n_keywords)
        self.finder_mode = "llm"
        return self.to_saged_data()

    @ignore_future_warnings
    def wiki_embeddings(self, keyword=None,
                        n_keywords=40, embedding_model='paraphrase-Mpnet-base-v2',
                        language='en', max_adjustment = 150,
                        user_agent='SAGED-bias (contact@holisticai.com)'):
        if not keyword:
            keyword = self.concept

        # Search Wikipedia for the keyword
        print('Initiating the embedding model...')
        model = SentenceTransformer(embedding_model)
        try:
            page_content = search_wikipedia(keyword, language, user_agent)[0].text
        except AttributeError as e:
            raise AttributeError(f"Page not Found: {e}")

        if isinstance(page_content, str) and page_content.startswith("No Wikipedia page found"):
            return page_content

        # Tokenize the Wikipedia page content
        tokens = re.findall(r'\b\w+\b', page_content.lower())

        # Get unique tokens
        unique_tokens = list(set(tokens))

        # Get embeddings for unique tokens
        embeddings = {}
        token_embeddings = model.encode(unique_tokens, show_progress_bar=True)
        for token, embedding in zip(unique_tokens, token_embeddings):
            embeddings[token] = embedding

        keyword_embedding = model.encode([keyword.lower()], show_progress_bar=True)[0]
        similarities = {}
        for token, embedding in tqdm(embeddings.items(), desc="Calculating similarities"):
            similarities[token] = util.pytorch_cos_sim(keyword_embedding, embedding).item()

        ADDITIONAL_ITEMS = 20

        # Ensure n_keywords is non-negative and within valid range. Adjusts n_keywords accordingly
        if max_adjustment > 0 and n_keywords + max_adjustment > len(similarities) / 2:
            n_keywords = max(int(len(similarities) / 2) - max_adjustment - 1, 0)

        # Sort tokens by similarity score and select top candidates
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        # Ensure not to exceed the available number of items
        num_items_to_select = min(len(sorted_similarities), n_keywords * 2 + ADDITIONAL_ITEMS)
        similar_words_first_pass = sorted_similarities[:num_items_to_select]

        # Convert to dictionary
        words_dict = dict(similar_words_first_pass)

        # Construct non-containing set
        non_containing_set = construct_non_containing_set(words_dict.keys())

        # Filter based on non-containing set
        similar_words_dict = {k: v for k, v in words_dict.items() if k in non_containing_set}

        # Select top keywords
        self.keywords = sorted(similar_words_dict, key=lambda k: similar_words_dict[k], reverse=True)[
                        :min(n_keywords, len(similar_words_dict))]

        # Set mode and return processed data
        self.finder_mode = "embedding"
        return self.to_saged_data()


class SourceFinder:
    def __init__(self, keyword_saged_data, source_tag='default'):
        assert isinstance(keyword_saged_data, saged_data), "You need an saged_data as an input."
        keyword_data_tier = keyword_saged_data.data_tier
        assert saged_data.tier_order[keyword_data_tier] >= saged_data.tier_order['keywords'], "You need an ata with " \
                                                                                        "data_tier higher than " \
                                                                                        "scrap_area. "
        self.concept = keyword_saged_data.concept
        self.domain = keyword_saged_data.domain
        self.data = keyword_saged_data.data
        self.source_finder = []
        self.source_tage = source_tag
        self.source_type = 'unknown'
        self.use_database = keyword_saged_data.use_database
        self.database_config = keyword_saged_data.database_config

    def to_saged_data(self):

        formatted_source_finder = [{
            "source_tag": self.source_tage,
            "source_type": self.source_type,
            "source_specification": self.source_finder
        }]

        self.data[0]["concept_shared_source"] = formatted_source_finder
        source_finder = saged_data.create_data(concept=self.concept, domain=self.domain, data_tier='source_finder',
                                         data=self.data)
        source_finder.use_database = self.use_database
        source_finder.database_config = self.database_config

        return source_finder

    @ignore_future_warnings
    def wiki(self, top_n=5, bootstrap_url=None, language='en',
             user_agent='SAGED-bias (contact@holisticai.com)', scrape_backlinks=0):
        """
        Main function to search Wikipedia for a topic and find related pages.
        """

        def get_related_forelinks(topic, page, wiki_wiki, max_depth=1, current_depth=0, visited=None, top_n=50):
            """
            Recursively get related pages up to a specified depth.

            Args:
            - topic (str): The main topic to start the search from.
            - page (Wikipedia page object): The Wikipedia page object of the main topic.
            - wiki_wiki (Wikipedia): The Wikipedia API instance.
            - max_depth (int): Maximum depth to recurse.
            - current_depth (int): Current depth of the recursion.
            - visited (set): Set of visited pages to avoid loops.
            - top_n (int): Number of top forelinks to retrieve based on relevance.

            Returns:
            - list: A list of URLs of related forelinks.
            """
            links = page.links
            related_pages = []

            if visited is None:
                visited = set()
                related_pages.append(page.fullurl)

            visited.add(page.title)

            title_list = list(links.keys())
            if len(title_list) > top_n:
                title_list = find_similar_keywords('paraphrase-MiniLM-L6-v2', topic, title_list, top_n)

            for link_title in tqdm(title_list, desc=f"Depth {current_depth + 1}/{max_depth}"):
                if link_title not in visited:
                    # try:
                    link_page = wiki_wiki.page(link_title)
                    if link_page.exists():
                        related_pages.append(link_page.fullurl)
                        visited.add(link_title)
                        if current_depth + 1 < max_depth:
                            # Pass `top_n` down in the recursive call
                            related_pages.extend(
                                get_related_forelinks(topic, link_page, wiki_wiki, max_depth, current_depth + 1,
                                                      visited,
                                                      top_n))
                    # except Exception as e:
                    #     print(f"Error with page {link_title}: {e}")

            return related_pages

        def get_related_backlinks(topic, page, wiki_wiki, max_depth=1, current_depth=0, visited=None, top_n=50):
            """
            Recursively get related backlinks up to a specified depth.

            Args:
            - topic (str): The main topic to start the search from.
            - page (Wikipedia page object): The Wikipedia page object of the main topic.
            - wiki_wiki (Wikipedia): The Wikipedia API instance.
            - max_depth (int): Maximum depth to recurse.
            - current_depth (int): Current depth of the recursion.
            - visited (set): Set of visited pages to avoid loops.
            - top_n (int): Number of top backlinks to retrieve based on relevance.

            Returns:
            - list: A list of URLs of related backlinks.
            """
            backlinks = page.backlinks
            related_pages = []

            if visited is None:
                visited = set()
                related_pages.append(page.fullurl)

            visited.add(page.title)

            title_list = list(backlinks.keys())
            if len(title_list) > top_n:
                title_list = find_similar_keywords('paraphrase-MiniLM-L6-v2', topic, title_list, top_n)

            for link_title in tqdm(title_list, desc=f"Depth {current_depth + 1}/{max_depth}"):
                if link_title not in visited:
                    try:
                        link_page = wiki_wiki.page(link_title)
                        if link_page.exists():
                            related_pages.append(link_page.fullurl)
                            visited.add(link_title)
                            if current_depth + 1 < max_depth:
                                # Pass `top_n` down in the recursive call
                                related_pages.extend(
                                    get_related_backlinks(topic, link_page, wiki_wiki, max_depth, current_depth + 1,
                                                          visited,
                                                          top_n))
                    except Exception as e:
                        print(f"Error with page {link_title}: {e}")

            return related_pages

        topic = self.concept

        print(f"Searching Wikipedia for topic: {topic}")
        main_page, wiki_wiki = search_wikipedia(topic, language=language, user_agent=user_agent)
        if top_n == 0:
            main_page_link = main_page.fullurl
            self.source_finder = [main_page_link]
            self.source_type = 'wiki_urls'
            return self.to_saged_data()

        if isinstance(main_page, str):
            return self.to_saged_data()
        else:
            print(f"Found Wikipedia page: {main_page.title}")
            print(f"Searching similar forelinks for {topic}")
            related_pages = get_related_forelinks(topic, main_page, wiki_wiki, max_depth=1, top_n=top_n)
            if scrape_backlinks > 0:
                print(f"Searching similar backlinks for {topic}")
                related_backlinks = get_related_backlinks(topic, main_page, wiki_wiki, max_depth=1, top_n=scrape_backlinks)
                related_pages.extend(related_backlinks)
            self.source_finder = list(set(related_pages))
            self.source_type = 'wiki_urls'
            return self.to_saged_data()

    def local(self, directory_path, direct_path_list=None):
        """Find text files either from a directory or use provided direct path list.
        
        Args:
            directory_path (str): Path to directory containing text files
            direct_path_list (list, optional): List of direct file paths to use instead of searching directory
        """
        if direct_path_list is not None and len(direct_path_list) > 0:
            # Use provided direct path list
            file_paths =  direct_path_list
        else:
            # Use glob to find all text files in the directory and its subdirectories
            text_files = glob.glob(os.path.join(directory_path, '**/*.txt'), recursive=True)
            file_paths = [file_path.replace('\\', '/') for file_path in text_files]
            
        self.source_finder = file_paths
        self.source_type = 'local_paths'
        return self.to_saged_data()


class Scraper:
    def __init__(self, source_finder_saged_data):
        assert isinstance(source_finder_saged_data, saged_data), "You need an saged_data with data_tier higher than source_finder."
        keyword_data_tier = source_finder_saged_data.data_tier
        assert saged_data.tier_order[keyword_data_tier] >= saged_data.tier_order['source_finder'], "You need an saged_data with " \
                                                                                          "data_tier higher than " \
                                                                                          "scrap_area. "
        self.concept = source_finder_saged_data.concept
        self.domain = source_finder_saged_data.domain
        self.data = source_finder_saged_data.data
        self.source_finder = source_finder_saged_data.data[0]["concept_shared_source"]
        self.keywords = self.data[0]["keywords"].keys()
        self.extraction_expression = r'(?<=\.)\s+(?=[A-Z])|(?<=\?")\s+|(?<=\.")\s+'  # Regex pattern to split sentences
        self.source_tag = 'default'
        self.use_database = source_finder_saged_data.use_database
        self.database_config = source_finder_saged_data.database_config

    @ignore_future_warnings
    def to_saged_data(self):
        scraped_sentences = saged_data.create_data(concept=self.concept, domain=self.domain,
                                                 data_tier='scraped_sentences',
                                                 data=self.data)
        scraped_sentences.use_database = self.use_database
        scraped_sentences.database_config = self.database_config
        return scraped_sentences

    @ignore_future_warnings
    def scrape_in_page_for_wiki_with_buffer_files(self):
        url_links = []
        source_tags_list = []
        for sa_dict in self.source_finder:
            if sa_dict["source_type"] == "wiki_urls":
                url_links.extend(sa_dict["source_specification"])
                source_tags_list.extend([sa_dict["source_tag"]] * len(sa_dict["source_specification"]))

        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Dictionary to store temporary files for each keyword
            temp_files = {}
            
            for url, source_tag in tqdm(zip(url_links, source_tags_list), desc='Scraping through URL', unit='url',
                                        total=min(len(url_links), len(source_tags_list))):
                url_results = []
                source_tag_buffer = []
                
                for keyword in tqdm(self.keywords, desc='Scraping in page', unit='keyword'):
                    # Fetch the HTML content of the URL
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Find all text elements in the HTML
                    text_elements = soup.find_all(['p', 'caption', 'figcaption'])

                    # Compile regex pattern to match keywords
                    keyword_regex = re.compile(r'\b(' + '|'.join([keyword]) + r')\b', re.IGNORECASE)

                    # Iterate through each text element
                    for element in text_elements:
                        # Remove references like '[42]' and '[page needed]'
                        clean_text = re.sub(r'\[\d+\]|\[.*?\]', '', element.get_text())

                        # Split text into sentences
                        sentences = re.split(self.extraction_expression, clean_text)

                        # Check each sentence for the keyword
                        for sentence in sentences:
                            if len(sentence.split()) >= 6 and keyword_regex.search(sentence):
                                url_results.append(sentence.strip())
                                source_tag_buffer.append(source_tag)

                    # Create temporary files for this keyword if they don't exist
                    if keyword not in temp_files:
                        temp_files[keyword] = {
                            'sentences': tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False),
                            'source_tags': tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
                        }

                    # Write results to temporary files
                    for sentence in url_results:
                        temp_files[keyword]['sentences'].write(f'{sentence}\n')
                    for tag in source_tag_buffer:
                        temp_files[keyword]['source_tags'].write(f'{tag}\n')

            # Process results for each keyword
            for keyword in self.keywords:
                if keyword in temp_files:
                    # Reset file pointers to beginning
                    temp_files[keyword]['sentences'].seek(0)
                    temp_files[keyword]['source_tags'].seek(0)
                    
                    # Read results
                    sentences = [line.strip() for line in temp_files[keyword]['sentences']]
                    source_tags = [line.strip() for line in temp_files[keyword]['source_tags']]
                    
                    # Create aggregated results
                    aggregated_results_with_source_tag = list(zip(sentences, source_tags))
                    
                    # Update the data structure
                    if "scraped_sentences" in self.data[0]["keywords"][keyword]:
                        self.data[0]["keywords"][keyword]["scraped_sentences"] = aggregated_results_with_source_tag
                    else:
                        self.data[0]["keywords"][keyword]["scraped_sentences"] = aggregated_results_with_source_tag
                    
                    # Close and cleanup temporary files
                    temp_files[keyword]['sentences'].close()
                    temp_files[keyword]['source_tags'].close()
                    os.unlink(temp_files[keyword]['sentences'].name)
                    os.unlink(temp_files[keyword]['source_tags'].name)

        return self.to_saged_data()

    @ignore_future_warnings
    def scrape_local_with_buffer_files(self, use_database=False, database_config=None):
        file_paths = []
        source_tags_list = []
        for sa_dict in self.source_finder:
            if sa_dict["source_type"] == "local_paths":
                file_paths.extend(sa_dict["source_specification"])
                source_tags_list.extend([sa_dict["source_tag"]] * len(sa_dict["source_specification"]))

        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Dictionary to store temporary files for each keyword
            temp_files = {}
            
            for file_path, source_tag in tqdm(zip(file_paths, source_tags_list), desc='Scraping through local files',
                                            unit='file', total=min(len(file_paths), len(source_tags_list))):
                path_results = []
                source_tag_buffer = []

                for keyword in tqdm(self.keywords, desc='Scraping in page', unit='keyword'):
                    # Read the file content using retrieve_txt
                    try:
                        text = saged_data.retrieve_txt(file_path, use_database, database_config)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
                        continue

                    # Clean the text by removing citations and other patterns within square brackets
                    text = text.replace('.\n', '. ').replace('\n', ' ')
                    clean_text = re.sub(r'\[\d+\]|\[.*?\]', '', text)

                    # Define a regex pattern for the keyword
                    keyword_regex = re.compile(re.escape(keyword), re.IGNORECASE)

                    # Split the cleaned text into sentences
                    sentences = re.split(r'(?<=\.)\s+(?=[A-Z])|(?<=\?")\s+|(?<=\.")\s+', clean_text)

                    # Extract desired sentences
                    for sentence in sentences:
                        if len(sentence.split()) >= 6 and keyword_regex.search(sentence):
                            path_results.append(sentence.strip())
                            source_tag_buffer.append(source_tag)

                    # Create temporary files for this keyword if they don't exist
                    if keyword not in temp_files:
                        temp_files[keyword] = {
                            'sentences': tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False),
                            'source_tags': tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
                        }

                    # Write results to temporary files
                    for sentence in path_results:
                        temp_files[keyword]['sentences'].write(f'{sentence}\n')
                    for tag in source_tag_buffer:
                        temp_files[keyword]['source_tags'].write(f'{tag}\n')

            # Process results for each keyword
            for keyword in self.keywords:
                if keyword in temp_files:
                    # Reset file pointers to beginning
                    temp_files[keyword]['sentences'].seek(0)
                    temp_files[keyword]['source_tags'].seek(0)
                    
                    # Read results
                    sentences = [line.strip() for line in temp_files[keyword]['sentences']]
                    source_tags = [line.strip() for line in temp_files[keyword]['source_tags']]
                    
                    # Create aggregated results
                    aggregated_results_with_source_tag = list(zip(sentences, source_tags))
                    
                    # Update the data structure
                    if "scraped_sentences" in self.data[0]["keywords"][keyword]:
                        self.data[0]["keywords"][keyword]["scraped_sentences"] = aggregated_results_with_source_tag
                    else:
                        self.data[0]["keywords"][keyword]["scraped_sentences"] = aggregated_results_with_source_tag
                    
                    # Close and cleanup temporary files
                    temp_files[keyword]['sentences'].close()
                    temp_files[keyword]['source_tags'].close()
                    os.unlink(temp_files[keyword]['sentences'].name)
                    os.unlink(temp_files[keyword]['source_tags'].name)

        return self.to_saged_data()