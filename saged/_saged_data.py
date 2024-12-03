import json
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

tqdm.pandas()


class SAGEDData:

    tier_order = {value: index for index, value in
                  enumerate(['keywords', 'source_finder', 'scraped_sentences', 'split_sentences', 'questions'])}

    default_source_item = {
        "source_tag": "default",
        "source_type": "unknown",
        "source_specification": []
    }

    default_keyword_metadata = {
        "keyword_type": "sub-concepts",
        "keyword_provider": "manual",
        "targeted_source": [{
            "source_tag": "default",
            "source_type": "unknown",
            "source_specification": []
        }],
        "scrap_mode": "in_page",
        "scrap_shared_area": "Yes"
    }


    def __init__(self, domain, category, data_tier, file_name=None):
        self.domain = domain
        self.category = category
        self.data_tier = data_tier
        assert data_tier in ['keywords', 'source_finder', 'scraped_sentences',
                             'split_sentences', 'questions'], "Invalid data tier. Choose from 'keywords', 'source_finder', 'scraped_sentences', 'split_sentences', 'questions'."
        # self.tier_order = {value: index for index, value in enumerate(['keywords', 'source_finder', 'scraped_sentences', 'split_sentences'])}
        self.file_name = file_name
        self.data = [{
            "category": self.category,
            "domain": self.domain}]

    @staticmethod
    def check_format(data_tier=None, data=None, source_finder_only=False):

        def check_source_finder_format(source_finder):
            assert isinstance(source_finder,
                              list), "source_finder should be a list of Sources dictionary"
            for source_finder in source_finder:
                assert isinstance(source_finder, dict), "Each item in the source_finder list should be a dictionary"
                source_finder_keys = {"source_tag", "source_type", "source_specification"}
                assert source_finder_keys == set(
                    source_finder.keys()), f"The Sources dictionary should contain only the keys {source_finder_keys}"
                assert source_finder['source_type'] in ['local_paths', 'wiki_urls',
                                                         'general_links', 'unknown'], "source_type should be either 'local_paths', 'wiki_urls','general_links', or 'unknown'."
                assert isinstance(source_finder['source_specification'],
                                  list), "source_specification should be a list of URLs or paths"

        if source_finder_only:
            return check_source_finder_format

        assert data_tier in ['keywords', 'source_finder', 'scraped_sentences',
                             'split_sentences', 'questions'], "Invalid data tier. Choose from 'keywords', 'source_finder', 'scraped_sentences', 'split_sentences', 'questions'."
        if data_tier in ['keywords', 'source_finder', 'scraped_sentences']:
            assert isinstance(data, list), "Data should be a list of dictionaries."
            for item in data:
                assert isinstance(item, dict), "Each item in the list should be a dictionary."
                assert 'keywords' in item, "Each dictionary should have a 'keywords' key."
                assert 'domain' in item, "Each dictionary should have a 'domain' key."
                assert 'category' in item, "Each dictionary should have a 'category' key."

                if data_tier in ['source_finder', 'scraped_sentences']:
                    assert 'category_shared_source' in item, "Each dictionary should have a 'category_shared_source' key"
                    source_finder = item['category_shared_source']
                    check_source_finder_format(source_finder)

                # check keywords format
                keywords = item['keywords']
                assert isinstance(keywords, dict), "keywords should be a dictionary"
                for k, v in keywords.items():
                    assert isinstance(v, dict), f"The value of keyword '{k}' should be a dictionary"
                    required_keys = {"keyword_type", "keyword_provider", "scrap_mode",
                                     "scrap_shared_area"}
                    if data_tier == 'scraped_sentences':
                        required_keys.add('scraped_sentences')
                        assert isinstance(v['scraped_sentences'],
                                          list), "scraped_sentences should be a list of sentences"


                    # check targeted_source format
                    if 'targeted_source' in v.keys():
                        required_keys.add('targeted_source')
                        assert required_keys == set(
                            v.keys()), f"The keywords dictionary of '{k}' should contain only the keys {required_keys} or with an additional 'targeted_source' key."
                        required_keys.remove('targeted_source')
                        check_source_finder_format(v['targeted_source'])
                    else:
                        assert required_keys == set(
                            v.keys()), f"The keywords dictionary of '{k}' should contain only the keys {required_keys}."

        elif data_tier == 'split_sentences':
            assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
            assert 'keyword' in data.columns, "DataFrame must contain 'keyword' column"
            assert 'category' in data.columns, "DataFrame must contain 'category' column"
            assert 'domain' in data.columns, "DataFrame must contain 'domain' column"
            assert 'prompts' in data.columns, "DataFrame must contain 'prompts' column"
            assert 'baseline' in data.columns, "DataFrame must contain 'baseline' column"

        elif data_tier == 'questions':
            print("IN HERE")
            assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
            assert 'keyword' in data.columns, "DataFrame must contain 'keyword' column"
            assert 'category' in data.columns, "DataFrame must contain 'category' column"
            assert 'domain' in data.columns, "DataFrame must contain 'domain' column"
            assert 'questions' in data.columns, "DataFrame must contain 'questions' column"
            assert 'original_answer' in data.columns, "DataFrame must contain 'original_answer' column"
            

    @classmethod
    def load_file(cls, domain, category, data_tier, file_path):
        instance = cls(domain, category, data_tier, file_path)
        try:
            if data_tier == 'split_sentences' or data_tier == 'questions':
                instance.data = pd.read_csv(file_path)
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                instance.data = data
            cls.check_format(data_tier, instance.data)
        except (IOError, json.JSONDecodeError, AssertionError) as e:
            print(f"Error loading or validating file {file_path}: {e}")
            return None

        return instance

    @classmethod
    def create_data(cls, domain, category, data_tier, data = None):
        instance = cls(domain, category, data_tier)

        if data is None:
            if data_tier == 'keywords':
                instance.data = [{
                    "category": category,
                    "domain": domain,
                    "keywords": {}
                }]
                return instance
            elif data_tier == 'source_finder':
                instance.data = [{
                    "category": category,
                    "domain": domain,
                    "category_shared_source": [cls.default_source_item],
                    "keywords": {}
                }]
                return instance
            elif data_tier == 'scraped_sentences':
                instance.data = [{
                    "category": category,
                    "domain": domain,
                    "category_shared_source": [cls.default_source_item],
                    "keywords": {}
                }]
                return instance
            elif data_tier == 'split_sentences':
                instance.data = pd.DataFrame(columns=['keyword', 'category', 'domain', 'prompts', 'baseline', 'source_tag'])
                return instance

        try:
            instance.data = data
            cls.check_format(data_tier, instance.data)
            return instance
        except (IOError, json.JSONDecodeError, AssertionError) as e:
            print(f"Error loading or validating data: {e}")
            return None

    def show(self, mode='short', keyword=None, data_tier=None):

        if SAGEDData.tier_order[data_tier] > SAGEDData.tier_order[self.data_tier]:
            print(f"Data in data tier '{data_tier}' is not available in the current data tier {self.data_tier}.")
            return

        if data_tier == 'keywords':
            if mode == 'short':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    keywords = ", ".join(item['keywords'].keys())
                    print(f"  Keywords: {keywords}")
            elif mode == 'metadata':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    for keyword, metadata in item['keywords'].items():
                        print(f"  Keyword: {keyword}, Metadata: {metadata}")
        elif data_tier == 'source_finder':
            if mode == 'short':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    for source_list in item['category_shared_source']:
                        print(f"  Sources: {source_list['source_specification']}")
            if mode == 'details':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    print(f"  Sources: {item['category_shared_source']}")
                    for keyword, metadata in item['keywords'].items():
                        print(f"  Keyword: {keyword}, targeted_source: {metadata.get('targeted_source')}")
        elif data_tier == 'scraped_sentences':
            if keyword == None:
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    for source_list in item['category_shared_source']:
                        print(f"  Sources: {source_list['source_specification']}")
                    for keyword, metadata in item['keywords'].items():
                        sentences = [i for i, _ in metadata.get('scraped_sentences')]
                        print(f"  Keyword '{keyword}' sentences: {sentences}")
        elif data_tier == 'split_sentences':
            print("Split sentences are in a DataFrame")
            print(self.data)
        elif data_tier == 'questions':
            print("Questions are in a DataFrame")
            print(self.data)

    def remove(self, entity, data_tier=None, keyword=None, removal_range='all'):

        def remove_element_from_list(main_list, sublist):
            for element in sublist:
                if element in main_list:
                    main_list.remove(element)
                    break
            return main_list

        if data_tier is None:
            data_tier = self.data_tier
        if not self.data:
            print("No data to modify.")
            return
        if SAGEDData.tier_order[data_tier] > SAGEDData.tier_order[self.data_tier]:
            print(f"Data tier '{data_tier}' is not available in the current data.")
            return

        if data_tier == 'keywords':
            for item in self.data:
                if entity in item['keywords']:
                    del item['keywords'][entity]
                    print(f"Keyword '{entity}' removed from the data.")

        if data_tier == 'source_finder' and removal_range == 'all':
            if isinstance(entity, dict):
                entity = [entity]

            for item in self.data:
                for sa_dict in item['category_shared_source']:
                    sa_dict['source_specification'] = remove_element_from_list(sa_dict['source_specification'],
                                                                                   entity)
                    print(f"Sources '{entity}' removed from the {sa_dict}.")
                for kw_dict in item['keywords']:
                    for sa_dict in kw_dict['targeted_source']:
                        sa_dict['source_specification'] = remove_element_from_list(
                            sa_dict['source_specification'], entity)
                        print(f"Sources '{entity}' removed from the {sa_dict}.")

        if data_tier == 'source_finder' and removal_range == 'targeted':
            # assert that the source_finder is in the right dictionary format
            if isinstance(entity, dict):
                entity = [entity]
            for index, source_finder_dict in enumerate(entity):
                # give source_tage if not provided
                if 'source_tag' not in source_finder_dict.keys():
                    entity[index]['source_tag'] = 'default'
            SAGEDData.check_format(source_finder_only=True)(entity)

            for item in self.data:
                if keyword:
                    for sa_index, sa_dict in enumerate(item['keywords'][keyword]['targeted_source']):
                        for sa_index_to_remove, sa_dict_to_remove in enumerate(entity):
                            if sa_dict['source_tag'] == sa_dict_to_remove['source_tag'] and \
                                    sa_dict['source_type'] == sa_dict_to_remove['source_type']:
                                remove_element_from_list(sa_dict['source_specification'],
                                                         sa_dict_to_remove['source_specification'])
                                print(
                                    f"Sources '{entity[sa_index_to_remove]['source_specification']}' removed from the data.")
                else:
                    for sa_index, sa_dict in enumerate(item['category_shared_source']):
                        for sa_index_to_remove, sa_dict_to_remove in enumerate(entity):
                            if sa_dict['source_tag'] == sa_dict_to_remove['source_tag'] and \
                                    sa_dict['source_type'] == sa_dict_to_remove['source_type']:
                                remove_element_from_list(sa_dict['source_specification'],
                                                         sa_dict_to_remove['source_specification'])
                                print(
                                    f"Sources '{sa_dict_to_remove['source_specification']}' removed from the data.")

        if data_tier == 'scraped_sentences':
            for item in self.data:
                for keyword, metadata in item['keywords'].items():
                    metadata['scraped_sentences'].remove(entity)
                    print(f"scraped sentence '{entity}' removed from the data.")
        if data_tier == 'split_sentences':
            print("Cannot remove from split sentences data, it is a DataFrame.")
        if data_tier == 'questions':
            print("Cannot remove from questions data, it is a DataFrame.")

    def add(self, keyword=None, source_finder=None, metadata=None, source_finder_target='common', data_tier=None):

        def merge_source_specifications(data):
            """
            Merges the source_specification lists for unique combinations of
            source_tag and source_type.

            Args:
                data (list): A list of dictionaries containing source_tag, source_type,
                             and source_specification.

            Returns:
                list: A list of merged dictionaries with unique source_tag and source_type.
            """
            merged_data = defaultdict(
                lambda: {"source_tag": "", "source_type": "", "source_specification": set()})

            for item in data:
                key = (item["source_tag"], item["source_type"])
                merged_data[key]["source_tag"] = item["source_tag"]
                merged_data[key]["source_type"] = item["source_type"]
                merged_data[key]["source_specification"].update(item["source_specification"])

            # Convert the sets back to lists
            result = []
            for value in merged_data.values():
                value["source_specification"] = list(value["source_specification"])
                result.append(value)

            return result

        if data_tier is None:
            data_tier = self.data_tier
        if SAGEDData.tier_order[data_tier] > SAGEDData.tier_order[self.data_tier]:
            print(f"Data tier '{data_tier}' is not available in the current data.")
            return self

        if data_tier == 'keywords':
            default_metadata = SAGEDData.default_keyword_metadata.copy()
            if metadata is None:
                metadata = default_metadata
            elif isinstance(metadata, dict):
                # Filter and update metadata based on default values
                filtered_metadata = {key: metadata.get(key, default_value) for key, default_value in
                                     default_metadata.items()}
                metadata = filtered_metadata
                targeted_source = metadata.get('targeted_source')
                SAGEDData.check_format(source_finder_only=True)(targeted_source)
            else:
                print("Metadata provided is not in the right dictionary format.")
                return self

            for index, item in enumerate(self.data):
                if 'keywords' not in item:
                    self.data[index]['keywords'] = {}
                self.data[index]['keywords'][keyword] = metadata
            return self

        if data_tier == 'source_finder':
            # assert that the source_finder is in the right dictionary format
            if isinstance(source_finder, dict):
                source_finder = [source_finder]
            for index, source_finder_dict in enumerate(source_finder):
                # give source_tage if not provided
                if 'source_tag' not in source_finder_dict.keys():
                    source_finder[index]['source_tag'] = 'default'
            SAGEDData.check_format(source_finder_only=True)(source_finder)

            for index, item in enumerate(self.data):
                if 'category_shared_source' not in item:
                    self.data[index]['category_shared_source'] = source_finder
                if source_finder_target == 'common':
                    self.data[index]['category_shared_source'].append(source_finder)
                    self.data[index]['category_shared_source'] = \
                        merge_source_specifications(self.data[index]['category_shared_source'])
                    SAGEDData.check_format(source_finder_only=True)(self.data[index]['category_shared_source'])

                elif source_finder_target == 'targeted':
                    assert keyword is not None, "Keyword must be provided to add targeted source."
                    self.data[index]['keywords'][keyword]['targeted_source'].append(source_finder)
                    self.data[index]['keywords'][keyword]['targeted_source'] = \
                        merge_source_specifications(self.data[index]['keywords'][keyword]['targeted_source'])
                    SAGEDData.check_format(source_finder_only=True)(
                        self.data[index]['keywords'][keyword]['targeted_source'])

            return self

        if data_tier == 'scraped_sentences':
            assert keyword is not None, "Keyword must be provided to add scraped sentences."
            for index, item in enumerate(self.data):
                self.data[index]['keywords'][keyword]['scraped_sentences'].append(source_finder)
                return self

        if data_tier == 'split_sentences':
            print("Cannot add to split sentences data, it is a DataFrame.")
            return self

    def save(self, file_path=None, domain_save=False, suffix=None):

        if self.data_tier == 'split_sentences' or self.data_tier == 'questions':
            if file_path is None:
                if domain_save:
                    file_name = f"{self.domain}_{self.data_tier}.csv"
                else:
                    file_name = f"{self.domain}_{self.category}_{self.data_tier}.csv"
                if suffix is not None:
                    file_name = f"{file_name[:-4]}_{suffix}.csv"
                default_path = os.path.join('data', 'customized', self.data_tier)
                os.makedirs(default_path, exist_ok=True)
                file_path = os.path.join(default_path, file_name)
            if isinstance(self.data, pd.DataFrame):
                self.data.to_csv(file_path, index=False)
                print(f"Data saved to {file_path}")
            else:
                print("Data is not in a DataFrame format.")
        else:
            # Generate default file name if not provided
            if file_path is None:
                file_name = f"{self.domain}_{self.category}_{self.data_tier}.json"
                # Ensure the default file path
                default_path = os.path.join('data', 'customized', self.data_tier)
                os.makedirs(default_path, exist_ok=True)
                file_path = os.path.join(default_path, file_name)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"Data saved to {file_path}")

    @classmethod
    def merge(cls, domain, merge_list, category = 'merged', saged_format = True):
        df = pd.DataFrame()
        for data_item in merge_list:
            assert isinstance(data_item, SAGEDData), "Data to merge should be of type saged_data."
            assert data_item.domain == domain, "Data to merge should have the same domain."
            assert data_item.data_tier == 'split_sentences', "Data to merge should be in split_sentences data tier."
            df = pd.concat([df, data_item.data], ignore_index=True)

        merged_data = SAGEDData.create_data(domain, category, 'split_sentences', df)

        if saged_format:
            return merged_data

        return df

    def sub_sample(self, sample=10, seed=42, clean=True, floor = False , saged_format = False):
        if not isinstance(self.data, pd.DataFrame):
            print("Cannot generate sub_sample from non-DataFrame data. You need to perform sentence split.")
            return

        if clean and 'keywords_containment' in self.data.columns:
            df = self.data
            df = df[df['keywords_containment'] == True]
            df = df.copy()  # Make a copy to avoid the SettingWithCopyWarning
            df.drop(['keywords_containment'], axis=1, inplace=True)
            self.data = df

        if floor:
            sample = min(sample, len(self.data))
        else:
           assert sample <= len(self.data), f"Sample size should be less than or equal to the data size {len(self.data)}."
        sample_data = self.data.sample(n=sample, random_state=seed).copy()
        self.data = sample_data

        if saged_format:
            return self

        return sample_data

    def model_generation(self, generation_function, generation_name='generated_output', saged_format = False):
        if not isinstance(self.data, pd.DataFrame):
            print("Cannot generate model output from non-DataFrame data. You need to perform sentence split.")
            return

        self.data[generation_name] = self.data['prompts'].progress_apply(generation_function)

        if saged_format:
            return self

        return self.data

