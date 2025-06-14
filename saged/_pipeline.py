from ._generator import ResponseGenerator
from ._extractor import FeatureExtractor
from ._diagnoser import DisparityDiagnoser as Analyzer
import pandas as pd
from ._saged_data import SAGEDData as saged
from ._scrape import KeywordFinder, SourceFinder, Scraper, check_generation_function
from ._assembler import PromptAssembler as PromptMaker
from ._utility import _update_configuration
from sqlalchemy import create_engine

class Pipeline:
    _branching_config_scheme = {}
    _concept_benchmark_config_scheme = {}
    _domain_benchmark_config_scheme = {}
    _branching_default_config = {}
    _concept_benchmark_default_config = {}
    _domain_benchmark_default_config = {}
    _analytics_config_scheme = {}
    _analytics_default_config = {}
    _database_config_scheme = {}
    _database_default_config = {}


    @classmethod
    def _set_config(cls):
        cls._database_config_scheme = {
            'use_database': None,
            'database_type': None,  # 'sql' or 'json'
            'database_connection': None,  # Connection string or path
        }
        
        cls._database_default_config = {
            'use_database': False,
            'database_type': 'json',
            'database_connection': 'data/customized/database',
        }

        cls._branching_config_scheme = {
            'branching_pairs': None,
            'direction': None,
            'source_restriction': None,
            'replacement_descriptor_require': None,
            'descriptor_threshold': None,
            'descriptor_embedding_model': None,
            'descriptor_distance': None,
            'replacement_description': None,
            'replacement_description_saving': None,
            'replacement_description_saving_location': None,
            'counterfactual_baseline': None,
            'generation_function': None,
        }
        cls._concept_benchmark_config_scheme = {
            'keyword_finder': {
                'require': None,
                'reading_location': None,
                'method': None,
                'keyword_number': None,
                'hyperlinks_info': None,
                'llm_info': None,
                'max_adjustment': None,
                'embedding_model': None,
                'saving': None,
                'saving_location': None,
                'manual_keywords': None,
            },
            'source_finder': {
                'require': None,
                'reading_location': None,
                'method': None,
                'local_file': None,
                'scrap_number': None,
                'saving': None,
                'saving_location': None,
                'scrape_backlinks': None,
            },
            'scraper': {
                'require': None,
                'reading_location': None,
                'saving': None,
                'method': None,  # This is related to the source_finder method,
                'saving_location': None},
            'prompt_assembler': {
                'require': None,
                'method': None,
                'generation_function': None,
                'keyword_list': None,
                'answer_check': None,
                'saving_location': None,
                'max_benchmark_length': None,
            },
        }
        cls._domain_benchmark_config_scheme = {
                                                  'categories': None,
                                                  'branching': None,
                                                  # If branching is False, then branching_config is not taken into account
                                                  'branching_config': None,
                                                  'shared_config': None,
                                                  'concept_specified_config': None,
                                                  'saving': None,
                                                  # If saving is False, then saving_location is not taken into account
                                                  'saving_location': None,
                                              }
        cls._branching_default_config = {
            'branching_pairs': 'not_all',
            'direction': 'both',
            'source_restriction': None,
            'replacement_descriptor_require': False,
            'descriptor_threshold': 'Auto',
            'descriptor_embedding_model': 'paraphrase-Mpnet-base-v2',
            'descriptor_distance': 'cosine',
            'replacement_description': {},
            'replacement_description_saving': True,
            'replacement_description_saving_location': f'data/customized/benchmark/replacement_description.json',
            'counterfactual_baseline': True,
            'generation_function': None,
        }
        cls._concept_benchmark_default_config = {
            'keyword_finder': {
                'require': True,
                'reading_location': 'default',
                'method': 'embedding_on_wiki',  # 'embedding_on_wiki' or 'llm_inquiries' or 'hyperlinks_on_wiki'
                'keyword_number': 7,  # keyword_number works for both embedding_on_wiki and hyperlinks_on_wiki
                'hyperlinks_info': [],
                # If hyperlinks_info is method chosen, can give more info... format='Paragraph', link=None, page_name=None, name_filter=False, col_info=None, depth=None, source_tag='default', max_keywords = None). col_info format is [{'table_num': value, 'column_name':List}]
                'llm_info': {},
                # If llm_inequiries is method chosen, can give more info... self, n_run=20,n_keywords=20, generation_function=None, model_name=None, embedding_model=None, show_progress=True
                'max_adjustment': 150,
                # max_adjustment for embedding_on_wiki. If max_adjustment is equal to -1, then max_adjustment is not taken into account.
                'embedding_model': 'paraphrase-Mpnet-base-v2',
                'saving': True,
                'saving_location': 'default',
                'manual_keywords': None,
            },
            'source_finder': {
                'require': True,
                'reading_location': 'default',
                'method': 'wiki',  # 'wiki' or 'local_files',
                'local_file': None,
                'scrap_number': 5,
                'saving': True,
                'saving_location': 'default',
                'scrape_backlinks': 0,
            },
            'scraper': {
                'require': True,
                'reading_location': 'default',
                'saving': True,
                'method': 'wiki',  # This is related to the source_finder method,
                'saving_location': 'default'},
            'prompt_assembler': {
                'require': True,
                'method': 'split_sentences',  # can also have "questions" as a method
                # prompt_assembler_generation_function and prompt_assembler_keyword_list are needed for questions
                'generation_function': None,
                # prompt_assembler_keyword_list must contain at least one keyword. The first keyword must be the keyword
                # of the original scraped data.
                'keyword_list': None,
                # User will enter False if they don't want their questions answer checked.
                'answer_check': False,
                'saving_location': 'default',
                'max_benchmark_length': 500,
            },
        }
        cls._domain_benchmark_default_config = {
            'categories': [],
            'branching': False,  # If branching is False, then branching_config is not taken into account
            'branching_config': cls._branching_default_config,
            'shared_config': cls._concept_benchmark_default_config,
            'concept_specified_config': {},
            'saving': True,  # If saving is False, then saving_location is not taken into account
            'saving_location': 'default',
        }
        cls._analytics_config_scheme = {
            "benchmark": None,
            "generation": {
                "require": None,
                "generate_dict": None,
                "generation_saving_location": None,
                "generation_list": None,
            },
            "extraction": {
                "feature_extractors": None,
                'extractor_configs': None,
                "calibration": None,
                "extraction_saving_location": None,
            },
            "analysis": {
                "specifications": None,
                "analyzers": None,
                "analyzer_configs": None,
                'statistics_saving_location': None,
                "disparity_saving_location": None,
            }
        }
        cls._analytics_default_config = {
            "generation": {
                "require": True,
                "generate_dict": {},
                "generation_saving_location": 'data/customized/' + '_' + 'sbg_benchmark.csv',
                "generation_list": [],
                "baseline": 'baseline',
            },
            "extraction": {
                "feature_extractors": [
                    'personality_classification',
                    'toxicity_classification',
                    'sentiment_classification',
                    'stereotype_classification',
                    'regard_classification'
                ],
                'extractor_configs': {},
                "calibration": True,
                "extraction_saving_location": 'data/customized/' + '_' + 'sbge_benchmark.csv',
            },
            "analysis": {
                "specifications": ['concept', 'source_tag'],
                "analyzers": ['mean', 'selection_rate', 'precision'],
                "analyzer_configs": {
                    'selection_rate': {'standard_by': 'mean'},
                    'precision': {'tolerance': 0.1}
                },
                'statistics_saving_location': 'data/customized/' + '_' + 'sbgea_statistics.csv',
                "disparity_saving_location": 'data/customized/' + '_' + 'sbgea_disparity.csv',
            }
        }

    @classmethod
    def config_helper(cls):
        pass

    @classmethod
    def build_concept_benchmark(cls, domain, demographic_label, config=None):
        cls._set_config()
        configuration = _update_configuration(
            cls._concept_benchmark_config_scheme.copy(),
            cls._concept_benchmark_default_config.copy(),
            config.copy())

        # Get database configuration
        database_config = _update_configuration(
            cls._database_config_scheme.copy(),
            cls._database_default_config.copy(),
            config.get('database_config', {}))

        # Create initial data with database config
        kw = saged.create_data(domain=domain, concept=demographic_label, data_tier='keywords',
                             use_database=database_config['use_database'],
                             database_config=database_config)

        # Unpacking keyword_finder section
        keyword_finder_config = configuration['keyword_finder']
        keyword_finder_require, keyword_finder_reading_location, keyword_finder_method, \
        keyword_finder_keyword_number, keyword_finder_hyperlinks_info, keyword_finder_llm_info, \
        keyword_finder_max_adjustment, keyword_finder_embedding_model, keyword_finder_saving, \
        keyword_finder_saving_location, keyword_finder_manual_keywords = (
            keyword_finder_config[key] for key in [
            'require', 'reading_location', 'method', 'keyword_number', 'hyperlinks_info',
            'llm_info', 'max_adjustment', 'embedding_model', 'saving', 'saving_location',
            'manual_keywords'
        ]
        )

        # Unpacking source_finder section
        source_finder_config = configuration['source_finder']
        source_finder_require, source_finder_reading_location, source_finder_method, \
        source_finder_local_file, source_finder_saving, source_finder_saving_location, \
        source_finder_scrap_area_number, source_finder_scrap_backlinks = (
            source_finder_config[key] for key in [
            'require', 'reading_location', 'method', 'local_file', 'saving',
            'saving_location', 'scrap_number', 'scrape_backlinks'
        ]
        )

        # Unpacking scraper section
        scraper_config = configuration['scraper']
        scraper_require, scraper_reading_location, scraper_saving, \
        scraper_method, scraper_saving_location = (
            scraper_config[key] for key in [
            'require', 'reading_location', 'saving', 'method', 'saving_location'
        ]
        )

        # Unpacking prompt_assembler section
        prompt_assembler_config = configuration['prompt_assembler']
        prompt_assembler_require, prompt_assembler_method, prompt_assembler_generation_function, \
        prompt_assembler_keyword_list, prompt_assembler_answer_check, prompt_assembler_saving_location, \
        prompt_assembler_max_sample_number = (
            prompt_assembler_config[key] for key in [
            'require', 'method', 'generation_function', 'keyword_list', 'answer_check',
            'saving_location', 'max_benchmark_length'
        ]
        )

        # check the validity of the configuration
        assert keyword_finder_method in ['embedding_on_wiki', 'llm_inquiries',
                                         'hyperlinks_on_wiki'], "Invalid keyword finder method. Choose either 'embedding_on_wiki', 'llm_inquiries', or 'hyperlinks_on_wiki'."
        if keyword_finder_method == 'llm_inquiries':
            assert 'generation_function' in keyword_finder_llm_info and keyword_finder_llm_info[
                'generation_function'] is not None, "generation function must be provided if llm_inquiries is chosen as the method"
            check_generation_function(keyword_finder_llm_info['generation_function'])
        assert source_finder_method in ['wiki',
                                        'local_files'], "Invalid scrap area finder method. Choose either 'wiki' or 'local_files'"
        assert scraper_method in ['wiki',
                                  'local_files'], "Invalid scraper method. Choose either 'wiki' or 'local_files'"
        assert source_finder_method == scraper_method, "source_finder_finder and scraper methods must be the same"
        assert prompt_assembler_method in ['split_sentences',
                                       'questions'], "Invalid prompt maker method. Choose 'split_sentences' or 'questions'"

        '''
        # make sure only the required loading is done
        if not scraper_require:
            source_finder_finder_require = False

        if not source_finder_finder_require:
            keyword_finder_require = False
        '''

        if keyword_finder_require:
            if keyword_finder_method == 'embedding_on_wiki':
                kw = KeywordFinder(domain=domain, concept=demographic_label).wiki_embeddings(
                    n_keywords=keyword_finder_keyword_number, embedding_model=keyword_finder_embedding_model,
                    max_adjustment=keyword_finder_max_adjustment).add(
                    keyword=demographic_label)
            elif keyword_finder_method == 'llm_inquiries':
                # format = self, n_run=20,n_keywords=20, generation_function=None, model_name=None, embedding_model=None, show_progress=True
                default_values = {'n_run': 20, 'n_keywords': 20, 'generation_function': None, 'model_name': None,
                                  'embedding_model': None, 'show_progress': True}

                # Create a new dictionary with only non-default fields
                for key in keyword_finder_llm_info:
                    if key in default_values:
                        default_values[key] = keyword_finder_llm_info[key]

                kw = KeywordFinder(domain=domain, concept=demographic_label).lm_inquiries(
                    **default_values).add(
                    keyword=demographic_label)

            # if manual keywords are provided, add them to the keyword finder
            if isinstance(keyword_finder_manual_keywords, list):
                for keyword in keyword_finder_manual_keywords:
                    kw = kw.add(keyword)

            if keyword_finder_saving:
                if keyword_finder_saving_location == 'default':
                    kw.save(use_database=database_config['use_database'], database_config=database_config)
                else:
                    kw.save(file_path=keyword_finder_saving_location, use_database=database_config['use_database'], database_config=database_config)

        elif (not keyword_finder_require) and isinstance(keyword_finder_manual_keywords, list):
            kw = saged.create_data(domain=domain, concept=demographic_label, data_tier='keywords', 
                                 use_database=database_config['use_database'], database_config=database_config)
            for keyword in keyword_finder_manual_keywords:
                kw = kw.add(keyword)

        elif source_finder_require and (keyword_finder_manual_keywords is None):
            filePath = ""
            if keyword_finder_reading_location == 'default':
                filePath = f'data/customized/keywords/{domain}_{demographic_label}_keywords.json'
                kw = saged.load_file(domain=domain, concept=demographic_label,
                                     file_path=filePath,
                                     data_tier='keywords',
                                     use_database=database_config['use_database'],
                                     database_config=database_config)
            else:
                filePath = keyword_finder_reading_location
                kw = saged.load_file(domain=domain, concept=demographic_label,
                                     file_path=filePath, data_tier='keywords',
                                     use_database=database_config['use_database'],
                                     database_config=database_config)

            if kw != None:
                print(f'Keywords loaded from {filePath}')
            else:
                raise ValueError(f"Unable to read keywords from {filePath}. Can't scrap area.")

        if source_finder_require:
            if source_finder_method == 'wiki':
                sa = SourceFinder(kw, source_tag='wiki').wiki(
                    top_n=source_finder_scrap_area_number, scrape_backlinks=source_finder_scrap_backlinks)
            elif source_finder_method == 'local_files':
                if source_finder_local_file == None:
                    raise ValueError(f"Unable to read keywords from {source_finder_local_file}. Can't scrap area.")
                sa = SourceFinder(kw, source_tag='local').local(source_finder_local_file)
            print('...Sources for Scraping located...')

            if source_finder_saving:
                if source_finder_saving_location == 'default':
                    sa.save(use_database=database_config['use_database'], database_config=database_config)
                else:
                    sa.save(file_path=source_finder_saving_location, use_database=database_config['use_database'], database_config=database_config)
        elif scraper_require:
            filePath = ""
            if source_finder_reading_location == 'default':
                filePath = f'data/customized/source_finder/{domain}_{demographic_label}_source_finder.json'
                sa = saged.load_file(domain=domain, concept=demographic_label,
                                     file_path=filePath,
                                     data_tier='source_finder',
                                     use_database=database_config['use_database'],
                                     database_config=database_config)
            else:
                filePath = source_finder_reading_location
                sa = saged.load_file(domain=domain, concept=demographic_label,
                                     file_path=source_finder_reading_location, data_tier='source_finder',
                                     use_database=database_config['use_database'],
                                     database_config=database_config)

            if sa != None:
                print(f'...Source info loaded from {filePath}...')
            else:
                raise ValueError(f"Unable to load Source info from {filePath}. Can't use scraper.")

        if scraper_require:
            if scraper_method == 'wiki':
                sc = Scraper(sa).scrape_in_page_for_wiki_with_buffer_files()
            elif scraper_method == 'local_files':
                sc = Scraper(sa).scrape_local_with_buffer_files()
            print('Scraped sentences completed.')

            if scraper_saving:
                if scraper_saving_location == 'default':
                    sc.save(use_database=database_config['use_database'], database_config=database_config)
                else:
                    sc.save(file_path=scraper_saving_location, use_database=database_config['use_database'], database_config=database_config)
        elif prompt_assembler_require:
            filePath = ""
            if scraper_reading_location == 'default':
                filePath = f'data/customized/scraped_sentences/{domain}_{demographic_label}_scraped_sentences.json'
                sc = saged.load_file(domain=domain, concept=demographic_label,
                                     file_path=filePath,
                                     data_tier='scraped_sentences',
                                     use_database=database_config['use_database'],
                                     database_config=database_config)
                print(
                    f'Scraped sentences loaded from data/customized/scraped_sentences/{domain}_{demographic_label}_scraped_sentences.json')
            else:
                filePath = scraper_reading_location
                sc = saged.load_file(domain=domain, concept=demographic_label, file_path=scraper_reading_location,
                                     data_tier='scraped_sentences',
                                     use_database=database_config['use_database'],
                                     database_config=database_config)
                print(f'Scraped sentences loaded from {scraper_reading_location}')

            if sc != None:
                print(f'Scraped loaded from {filePath}')
            else:
                raise ValueError(f"Unable to load scraped sentences from {filePath}. Can't make prompts.")

        pm_result = None
        if prompt_assembler_method == 'split_sentences' and prompt_assembler_require:
            pm = PromptMaker(sc)
            pm_result = pm.split_sentences()
        elif prompt_assembler_method == 'questions' and prompt_assembler_require:
            pm = PromptMaker(sc)
            pm_result = pm.make_questions(generation_function=prompt_assembler_generation_function,
                                          keyword_reference=prompt_assembler_keyword_list,
                                          answer_check=prompt_assembler_answer_check,
                                          max_questions=prompt_assembler_max_sample_number)
        if pm_result is None:
            raise ValueError(f"Unable to make prompts out of no scraped sentences")
        pm_result = pm_result.sub_sample(prompt_assembler_max_sample_number, floor=True,
                                         saged_format=True)  ### There is likely a bug
        if prompt_assembler_saving_location == 'default':
            pm_result.save(use_database=database_config['use_database'], database_config=database_config)
        else:
            pm_result.save(file_path=prompt_assembler_saving_location, use_database=database_config['use_database'], database_config=database_config)

        print(f'Benchmark building for {demographic_label} completed.')
        print('\n=====================================================\n')

        return pm_result

    @classmethod
    def build_benchmark(cls, domain, config=None):
        def _merge_concept_specified_configuration(domain_configuration):

            def _simple_update_configuration(default_configuration, updated_configuration):
                """
                Update the default configuration dictionary with the values from the updated configuration
                only if the keys already exist in the default configuration.

                Args:
                - default_concept_configuration (dict): The default configuration dictionary.
                - updated_configuration (dict): The updated configuration dictionary with new values.

                Returns:
                - dict: The updated configuration dictionary.
                """

                for key, value in updated_configuration.items():
                    if key in default_configuration.copy():
                        # print(f"Updating {key} recursively")
                        if isinstance(default_configuration[key], dict) and isinstance(value, dict):
                            # Recursively update nested dictionaries
                            default_configuration[key] = _simple_update_configuration(default_configuration[key].copy(),
                                                                                      value)
                        else:
                            # print(f"Skipping key: {key} due to type mismatch")
                            # Update the value for the key
                            default_configuration[key] = value
                return default_configuration

            specified_config = domain_configuration['concept_specified_config'].copy()
            domain_configuration = _simple_update_configuration(Pipeline._domain_benchmark_default_config.copy(),
                                                                domain_configuration)
            domain_configuration['shared_config'] = _simple_update_configuration(
                Pipeline._concept_benchmark_default_config .copy(), domain_configuration['shared_config'].copy())

            base_concept_config = {}
            for cat in domain_configuration['categories']:
                base_concept_config[cat] = domain_configuration['shared_config'].copy()

            # print('start ====================== \n\n')
            merge_concept_config = _simple_update_configuration(base_concept_config.copy(), specified_config.copy())
            # print(merge_concept_config)
            return merge_concept_config

        cls._set_config()
        concept_list = config['categories']
        concept_specified_configuration = _merge_concept_specified_configuration(config.copy())
        configuration = _update_configuration(
            cls._domain_benchmark_config_scheme.copy(),
            cls._domain_benchmark_default_config.copy(),
            config.copy())

        # Get database configuration
        database_config = _update_configuration(
            cls._database_config_scheme.copy(),
            cls._database_default_config.copy(),
            config.get('database_config', {}))

        # Create domain benchmark with database config
        domain_benchmark = saged.create_data(domain=domain, concept='all', data_tier='split_sentences',
                                           use_database=database_config['use_database'],
                                           database_config=database_config)
        for concept in concept_list:
            cat_result = cls.build_concept_benchmark(domain, concept, concept_specified_configuration[concept])
            print(f'Benchmark building for {concept} completed.')
            domain_benchmark = saged.merge(domain, [domain_benchmark, cat_result])

            if configuration['saving']:
                if configuration['saving_location'] == 'default':
                    domain_benchmark.save(use_database=database_config['use_database'],
                                        database_config=database_config)
                else:
                    domain_benchmark.save(file_path=configuration['saving_location'],
                                        use_database=database_config['use_database'],
                                        database_config=database_config)

        if configuration['branching']:
            empty_ss = saged.create_data(concept='merged', domain=domain, data_tier='scraped_sentences',
                                       use_database=database_config['use_database'],
                                       database_config=database_config)
            pmr = PromptMaker(empty_ss)
            pmr.output_df = domain_benchmark.data
            domain_benchmark = pmr.branching(branching_config=configuration['branching_config'])
            if configuration['saving']:
                if configuration['saving_location'] == 'default':
                    domain_benchmark.save(suffix='branching',
                                        use_database=database_config['use_database'],
                                        database_config=database_config)
                else:
                    domain_benchmark.save(file_path=configuration['saving_location'],
                                        use_database=database_config['use_database'],
                                        database_config=database_config)

        return domain_benchmark

    @classmethod
    def run_benchmark(cls, config, domain='unspecified'):
        cls._set_config()
        v = _update_configuration(
            cls._analytics_config_scheme.copy(),
            cls._analytics_default_config.copy(),
            config.copy())

        # Get database configuration
        database_config = _update_configuration(
            cls._database_config_scheme.copy(),
            cls._database_default_config.copy(),
            config.get('database_config', {}))

        def save_to_database_or_file(df, location, suffix=None):
            if database_config['use_database']:
                engine = create_engine(database_config['database_connection'])
                table_name = location.replace('.csv', '')
                if suffix:
                    table_name = f"{table_name}_{suffix}"
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                print(f"Data saved to database table {table_name}")
            else:
                if suffix:
                    location = location.replace('.csv', f'_{suffix}.csv')
                df.to_csv(location, index=False)
                print(f"Data saved to {location}")

        if v['generation']['require']:
            gen = ResponseGenerator(v['benchmark'])

            for name, gf in v['generation']['generate_dict'].items():
                gen.generate(gf, generation_name=name, save_path=v['generation']['generation_saving_location'])
            sbg_benchmark = gen.benchmark.copy()
            save_to_database_or_file(sbg_benchmark, v['generation']['generation_saving_location'])

            generation_list = list(v['generation']['generate_dict'].keys())
            glb = ['baseline'] + generation_list.copy()
        else:
            sbg_benchmark = v['benchmark']
            generation_list = v['generation']['generation_list']
            glb = ['baseline'] + generation_list

        fe = FeatureExtractor(sbg_benchmark, generations=glb, calibration=v['extraction']['calibration'])

        sbge_benchmark = pd.DataFrame()
        for x in v['extraction']['feature_extractors']:
            try:
                method_to_call = getattr(fe, x)
                sbge_benchmark = method_to_call(**v['extraction']['extractor_configs'].get(x, {}))
            except AttributeError as e:
                print(f"Method {x} does not exist: {e}")
            except Exception as e:
                print(f"Error calling method {x}: {e}")
        save_to_database_or_file(sbge_benchmark, v['extraction']['extraction_saving_location'])
        raw_features = fe.classification_features + fe.cluster_features
        calibrated_features = fe.calibrated_features

        print(raw_features)
        print(calibrated_features)

        anas = []
        anas.append(Analyzer(sbge_benchmark.copy(), features=raw_features, generations=glb))
        if v['extraction']['calibration']:
            anas.append(
                Analyzer(sbge_benchmark.copy(), features=calibrated_features, generations=generation_list))

        for k, ana in enumerate(anas):
            ana.specifications = v['analysis']['specifications']
            for x in v['analysis']['analyzers']:
                try:
                    method_to_call = getattr(ana, x)
                    sbgea_benchmark = method_to_call(test=False, **v['analysis']['analyzer_configs'].get(x, {}))
                    if k == 0:
                        save_to_database_or_file(sbgea_benchmark, v['analysis']['statistics_saving_location'], x)
                    elif k == 1:
                        save_to_database_or_file(sbgea_benchmark, v['analysis']['statistics_saving_location'], f'calibrated_{x}')
                except AttributeError as e:
                    print(f"Method {x} does not exist: {e}")
                except Exception as e:
                    print(f"Error calling method {x}: {e}")
            df = ana.statistics_disparity()
            if k == 0:
                save_to_database_or_file(df, v['analysis']['disparity_saving_location'])
            elif k == 1:
                save_to_database_or_file(df, v['analysis']['disparity_saving_location'], 'calibrated')