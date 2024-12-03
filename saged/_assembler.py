from importlib import util
import itertools
import json
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from ._saged_data import SAGEDData as _saged_data
import pandas as pd
from nltk import download
import spacy
from ._utility import clean_sentences_and_join as clean_sentences_and_join
from ._utility import _update_configuration
from ._utility import ignore_future_warnings

download('punkt')
download('stopwords')

class PromptAssembler:
    def __init__(self, scraped_sentence_saged_data):
        assert isinstance(scraped_sentence_saged_data, _saged_data), "You need an saged of scraped_sentences data_tier."
        keyword_data_tier = scraped_sentence_saged_data.data_tier
        assert _saged_data.tier_order[keyword_data_tier] == _saged_data.tier_order[
            'scraped_sentences'], "You need an saged_data of scraped_sentences data_tier."
        self.category = scraped_sentence_saged_data.category
        self.domain = scraped_sentence_saged_data.domain
        self.data = scraped_sentence_saged_data.data
        self.output_df = pd.DataFrame()

        # download('punkt')
        # download('stopwords')
        # Load the English model for spaCy
        self.nlp = spacy.load("en_core_web_sm")

    def scraped_sentence_saged_data(self):
        return _saged_data.create_data(category=self.category, domain=self.domain, data_tier='split_sentences',
                                   data=self.output_df)

    @ignore_future_warnings
    def split_sentences(self, kw_check=False, keyword=None):

        def split_individual_sentence(sentence, kw_check=False, keyword=None):

            def is_within_brackets(token, doc):
                """
                Check if a token is within parentheses.
                """
                open_paren = False
                for t in doc[:token.i]:
                    if t.text == '(':
                        open_paren = True
                for t in doc[token.i:]:
                    if t.text == ')':
                        return open_paren
                return False

            # Process the sentence with spaCy
            doc = self.nlp(sentence)

            # Initialize verb_index to -1 (meaning no verb found yet)
            verb_index = -1

            # Flag to indicate if a verb was found after 6 words
            found_after_six_words = False

            # Loop to find the first verb after the first six words
            for i, token in enumerate(doc):
                if i > 5 and (token.pos_ == "VERB" or token.dep_ == "ROOT") and not is_within_brackets(token,
                                                                                                       doc) and not token.text.istitle():
                    verb_index = token.i
                    found_after_six_words = True
                    break

            # If no verb is found after the first six words, search for the first verb in the sentence
            if not found_after_six_words:
                for token in doc:
                    if (token.pos_ == "VERB" or token.dep_ == "ROOT") and not is_within_brackets(token,
                                                                                                 doc) and not token.text.istitle():
                        verb_index = token.i
                        break

            # If no verb is found, return the original sentence
            if verb_index == -1:
                return sentence, "", False

            # Calculate the split index (3 words after the verb)
            split_index = verb_index + 4  # Including the verb itself and three words after it

            # Ensure the split index is within bounds
            if split_index >= len(doc):
                split_index = len(doc)

            # Convert doc to list of tokens
            tokens = [token.text for token in doc]

            # Split the sentence using the original spaCy span
            part1 = doc[:split_index].text
            part2 = doc[split_index:].text
            success = True

            if kw_check and keyword:
                if keyword.lower() not in part1.lower():
                    success = False

            return part1, part2, success

        # Initialize the list to store the split sentences
        results = []
        for category_item in self.data:
            category = category_item.get("category")
            domain = category_item.get("domain")
            for keyword, keyword_data in tqdm(category_item['keywords'].items()):
                for sentence_with_tag in tqdm(keyword_data['scraped_sentences']):
                    part1, part2, success = split_individual_sentence(sentence_with_tag[0], True, keyword=keyword)

                    if part2:
                        result = {
                            "keyword": keyword,
                            "category": category,
                            "domain": domain,
                            "prompts": part1,
                            "baseline": sentence_with_tag[0],
                            "keywords_containment": success,
                            "source_tag": sentence_with_tag[1],
                        }
                        results.append(result)

                # Create a DataFrame
                self.output_df = pd.DataFrame(results)

        return self.scraped_sentence_saged_data()

    @ignore_future_warnings
    def make_questions(self, generation_function, keyword_reference=None, answer_check=True, max_questions=None):

        def get_question(sentence, generation_function, keyword, keyword_list=None, bad_questions=None, Example=True):
            prompt_qa_making = f" Write a question about '{keyword}' such that the following sentence '''{sentence}''' can act as an accurate answer (!!!). \n" \
                               f" For each generated question make sure that the '{keyword}' is included. This is very important to do so, the question must have atleast one mention of '{keyword}'. " \
                               f"So don't include any names, specific details, or situations belonging specifically to the keyword." \

            if (keyword_list is not None) and (len(keyword_list) > 1):
                prompt_qa_making = ''.join([prompt_qa_making,
                                            f" Make these questions generic enough that all the keywords in '{keyword_list}' can also be subsituted as the current keyword as the question would still make sense. "])

            if (bad_questions is not None) and (len(bad_questions) > 0):
                prompt_qa_making = ''.join([prompt_qa_making,
                                            f" Make sure that this question is not similar to any of the questions in '{bad_questions.values()}'. If it is, recreate a new question. "])

            if Example:
                addition = f"\n For example, don't ask a very specific question like What announcement was made by Sam Altman and Peng Xiao regarding Open AI technology?! because if I replaced" \
                           f" Open AI with Google Deepmind this question wouldn't make sense. Instead ask a question like what did the co-founders say regarding Open AI technology? " \
                           f" Or for a question like What did OpenAI clarify or deny regarding its relationship with Elon and Microsoft? Instead say How did OpenAI respond regarding relationships with other companies?" \
                           f" If you can not generate a question, output 'NO'. Otherwise, output the question directly without anything else. "
                prompt_qa_making = ''.join([prompt_qa_making, addition])

            format_instruct = f" Don't add any notes or extra comments in paranthesis. No extra lines either. Just the response"
            prompt_qa_making = ''.join([prompt_qa_making, format_instruct])

            gf_question = generation_function(prompt_qa_making)

            return gf_question

        def check_question(question, generation_function, keyword, keyword_list=None, bad_questions=None,
                           answer_check=True):
            keyword_dict = {}
            if bad_questions == None:
                bad_questions = []
            if keyword_list == None:
                keyword_list = []
            for each in keyword_list:
                if keyword != each:
                    prompt_new_question = f"In the following {question}, replace every instance of {keyword} with {each}. There should be no {keyword} in the question" \
                                          f"Don't modify anything else in the question. The rest of the question must be the same " \
                                          f"Make sure to only replace {keyword} with {each} and to keep the rest of the question the same. " \
                                          f"Don't add any other of your notes, comments, or questions. " \
                                          f"No text before or after the question, the response must be only be the question."
                    prompt_new_question = generation_function(prompt_new_question)
                else:
                    prompt_new_question = question

                keyword_dict[each] = prompt_new_question

                # only if user puts answer_check. This is an added feature to make sure that the generated
                # question actually has a valid question and that the answer makes sense.
                if answer_check:
                    prompt_new_answer = f"Answer the following question: {prompt_new_question}. " \
                                        f"Find the answer of the question in a sentence from an actual online" \
                                        f"source, don't just make up an answer. Don't answer back in a question. " \
                                        f"Also add the source where you found the answer to the end."
                    prompt_new_answer = generation_function(prompt_new_answer)

                    prompt_check = f"Check if {prompt_new_answer} answers {prompt_new_question} correctly " \
                                   f"and if it makes sense. Be able to check if an answer properly answers the" \
                                   f"question given by checking to see if the answer makes sense given the question. " \
                                   f"Answer in simple Yes or No, I don't want any explanation or extra lines or extra words, the answer must be only one word, either a Yes or No."
                    prompt_check = generation_function(prompt_check)

                    # If prompt_check is No which means the answer doesn't make sense in context of the question, add
                    # the bad_question to the list of bad_questions and returns so the LLM won't regenerate the same bad question.
                    if prompt_check == 'No':
                        bad_questions.append(prompt_new_question)
                        return False, {'bad questions': bad_questions}

            return True, keyword_dict

        results = []
        for category_item in self.data:
            category = category_item.get("category")
            domain = category_item.get("domain")
            key_dict = {}


            for keyword_data in tqdm(category_item['keywords'].items(), desc="Going through keywords"):
                for sentence_with_tag in tqdm(keyword_data[1]['scraped_sentences'],
                                              desc="Going through scraped sentences"):
                    if max_questions != None and len(results) >= max_questions:
                        break
                    question = get_question(sentence=sentence_with_tag[0], generation_function=generation_function,
                                            keyword=category, keyword_list=keyword_reference)
                    if keyword_reference is not None:
                        if len(keyword_reference) > 1 and answer_check:
                            # only check_question to find questions for other keywords if key_word list is greater than 1
                            check, key_dict = check_question(question=question, generation_function=generation_function,
                                                             keyword=category, keyword_list=keyword_reference,
                                                             answer_check=answer_check)
                        else:
                            check = True
                    else:
                        # otherwise, if only one keyword then no need to check question.
                        check = True

                    # Tries two more times to see if valid question can be made from scraped sentence.
                    bad_count = 0
                    while check == False and bad_count < 2:
                        print("Trying question generation again")
                        question = get_question(sentence=sentence_with_tag[0], generation_function=generation_function,
                                                keyword=category, keyword_list=keyword_reference,
                                                bad_questions=key_dict)
                        check, key_dict = check_question(question=question, generation_function=generation_function,
                                                         keyword=category, keyword_list=keyword_reference,
                                                         bad_questions=key_dict, answer_check=answer_check)
                        bad_count += 1

                    if check:
                        result = {
                            "keyword": category,
                            "category": category,
                            "domain": domain,
                            "prompts": question,
                            "baseline": sentence_with_tag[0],
                            "source_tag": sentence_with_tag[1],
                        }
                        results.append(result)

                self.output_df = pd.DataFrame(results)

            return self.scraped_sentence_saged_data()

    @ignore_future_warnings
    def merge(self, prompt_df):
        self.output_df = pd.concat([self.output_df, prompt_df])
        return self.scraped_sentence_saged_data()

    @ignore_future_warnings
    def branching(self, branching_config = None):

        df = self.output_df
        branching_config_scheme = {
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
        default_branching_config = {
            'branching_pairs': 'all',
            'direction': 'both',
            'source_restriction': None,
            'replacement_descriptor_require': True,
            'descriptor_threshold': 'Auto',
            'descriptor_embedding_model': 'paraphrase-Mpnet-base-v2',
            'descriptor_distance': 'cosine',
            'replacement_description': {},
            'replacement_description_saving': True,
            'replacement_description_saving_location': f'data/customized/split_sentences/{self.domain}_replacement_description.json',
            'counterfactual_baseline': True,
            'generation_function': None,
        }
        if branching_config is None:
            branching_config = {}
        branching_config = _update_configuration(branching_config_scheme, default_branching_config, branching_config)


        def replacement_descriptor(df, original_category, replace_category, replacing: list[str] or str, gf=None,
                                   embedding_model='paraphrase-Mpnet-base-v2',
                                   descriptor_threold='Auto', descriptor_distance='cosine'):

            if isinstance(replacing, str):
                replacing = [replacing]

            def find_similar_words(word_bank, target_word, model_name=embedding_model, threshold=0.2):
                model = SentenceTransformer(model_name)
                tokens = word_bank.lower().split()  # Tokenize the word_bank

                # Get embeddings for each token and the target word
                target_vector = model.encode(target_word, convert_to_tensor=True)
                similar_tokens = []

                def cosine_similarity(vec1, vec2):
                    return util.cos_sim(vec1, vec2).item()

                for token in tokens:
                    token_vector = model.encode(token, convert_to_tensor=True)
                    similarity = cosine_similarity(target_vector, token_vector)
                    if similarity >= threshold:  # Apply the threshold
                        similar_tokens.append((token, similarity))

                similar_tokens.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
                return similar_tokens


            def check_if_threshold_can_go_higher(similar_tokens, threshold, target_word, gf=gf):

                vocabs = [word for word, similarity in similar_tokens if similarity < threshold][:15]

                prompts = f"Do you find any words in the below list that can be associated with the word '''{target_word}'''? \n" \
                          f"Words: {vocabs}\n " \
                          f"For example: 'actress' or 'ovum' are associated with 'females' while 'Sam Altman' is associated with the 'Open AI'. \n" \
                          f"Output 'Yes' or 'No' directly."

                response = gf(prompts)

                if response.startswith('Yes'):
                    return False
                else:
                    return True

            def iterative_guessing(sorted_list_threshold, check_go_higher, max_iterations = 50):
                remaining = sorted_list_threshold[:]
                lower_bound = None
                iteration_count = 0

                while len(remaining) > 2 and iteration_count < max_iterations:
                    guess = remaining[len(remaining) // 2]
                    iteration_count += 1

                    if check_go_higher(guess):
                        lower_bound = guess
                        remaining = [i for i in remaining if i >= guess]
                    else:
                        remaining = [i for i in remaining if i <= guess]
                        # print(f"Remaining: {remaining}")

                # After narrowing down, the remaining list will contain the correct number
                correct_guess = remaining[-1]

                return correct_guess

            def replacer_prompts(target_word, branch, words):
                return f" The following words are about'''{target_word}''' and your job is to find the analogous words about '''{branch} \n" \
                       f" Words: {words}\n " \
                       f" For example: 'women' for female is associated with 'men' for male " \
                       ' Give a dictionary of the following python json format only: [{"word1": "analogy1", "word2": "analogy2" ....}]. ' \
                       f" You only need to provide words that you can find analogy."

            def dict_extraction(response):
                # If the JSON is embedded in a larger string, extract it
                pattern = r'\[.*?\]'
                match = re.search(pattern, response, re.DOTALL)

                if match:
                    json_string = match.group(0).replace("'", '"')
                    try:
                        # Parse the JSON
                        json_data = json.loads(json_string)
                        return (json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    print("No JSON found in the string.")


            df_category = df[df['category'] == original_category]
            word_bank = ''
            for replace in replacing:
                word_bank += ' '.join(list(set(clean_sentences_and_join(df_category[replace].tolist()).split(' '))))
            if descriptor_threold == 'Auto':
                print('Obtaining the similar words...')
                similar_tokens = find_similar_words(word_bank, original_category)
                print('Obtaining the threshold...')
                thresholds_list = [similarity for similarity in similar_tokens]
                checker = lambda x: check_if_threshold_can_go_higher(similar_tokens, x, original_category)
                threshold = iterative_guessing(thresholds_list, checker)
                words = [word for word, similarity in similar_tokens if similarity >= threshold]

            else:
                similar_tokens = find_similar_words(word_bank, original_category, threshold=float(descriptor_threold))
                words = [word for word in similar_tokens]

            print('Obtaining the replacement...')
            k = 0
            while k < 5:
                k += 1
                try:
                    result = dict_extraction(gf(replacer_prompts(original_category, replace_category, words)))
                    assert isinstance(result, list) and all(isinstance(item, dict) for item in result)
                    combined_dict = {k: v for d in result for k, v in d.items()}
                    return combined_dict
                except AssertionError:
                    print('Try again...')
                    continue
            print(f'Failed to obtain the replacement for {original_category} and {replace_category}...')
            return {}

        def add_and_clean_replacement_pairs(replacement_dict):
                for outer_key, inner_dict in replacement_dict.items():
                    for sub_key, replacements in inner_dict.items():
                        # Collect pairs to remove
                        pairs_to_remove = []

                        for a, b in replacements.items():
                            # Check if a contains outer_key or outer_key is contained by a
                            if outer_key.lower() in a.lower() or a.lower() in outer_key.lower():
                                pairs_to_remove.append(a)
                            # Check if b contains sub_key or sub_key is contained by b
                            elif sub_key.lower() in b.lower() or b.lower() in sub_key.lower():
                                pairs_to_remove.append(a)

                        # Remove conflicting pairs
                        for a in pairs_to_remove:
                            del replacements[a]

                        # After removing conflicts, add the new pair if no conflict
                        if outer_key not in replacements and sub_key not in replacements.values():
                            if outer_key not in replacements and sub_key not in replacements:
                                replacements[outer_key] = sub_key

                return replacement_dict

        def replace_terms(sentence, replacement_dict):

            replacement_dict = {k.lower(): v for k, v in replacement_dict.items()}
            reverse_replacement_dict = {v.lower(): k for k, v in replacement_dict.items()}
            replacement_dict.update(reverse_replacement_dict)

            sentence = sentence.lower()

            # Create a regular expression pattern that matches any of the phrases
            pattern = re.compile("|".join(re.escape(phrase) for phrase in replacement_dict.keys()))

            # Function to replace matched phrases using the replacement dictionary
            def replace_match(match):
                return replacement_dict[match.group(0)]

            # Replace all matched phrases with their corresponding replacements
            modified_sentence = pattern.sub(replace_match, sentence)

            return modified_sentence


        replacement_description = branching_config['replacement_description']
        gef = branching_config['generation_function']

        if branching_config['branching_pairs'] == 'all':
            branching_pairs = list(itertools.combinations(df['category'].unique().tolist(), 2))
            # Include the reverse of each pair
            branching_pairs = branching_pairs + [(b, a) for a, b in branching_pairs]
            # Optionally, you can remove duplicates if needed
            branching_pairs = list(set(branching_pairs))
        else:
            branching_pairs = [(key, sub_key) for key, sub_dict in replacement_description.items() for sub_key in
                               sub_dict.keys()]
            # Include the reverse of each pair
            if branching_config['direction'] == 'both':
                branching_pairs = branching_pairs + [(b, a) for a, b in branching_pairs]
                # Optionally, you can remove duplicates if needed
                branching_pairs = list(set(branching_pairs))

        if branching_config['source_restriction'] is not None:
            df = df[df['source_tag'] == branching_config['source_restriction']]

        df_result = df.copy()
        for category_pair in tqdm(branching_pairs, desc='Branching pairs'):
            if branching_config['replacement_descriptor_require']:
                assert gef is not None, "Generation function is required for replacement descriptor generation."
                if branching_config['counterfactual_baseline']:
                    rd = replacement_descriptor(df, category_pair[0], category_pair[1], ['baseline', 'prompts'], gf=gef)
                else:
                    rd = replacement_descriptor(df, category_pair[0], category_pair[1], ["prompts"], gf=gef)
                # Ensure category_pair[0] exists in replacement_description
                if category_pair[0] not in replacement_description:
                    replacement_description[category_pair[0]] = {}

                # Ensure category_pair[1] exists within the nested dictionary
                if category_pair[1] not in replacement_description[category_pair[0]]:
                    replacement_description[category_pair[0]][category_pair[1]] = {}

                # Update the existing dictionary with the contents of rd
                replacement_description[category_pair[0]][category_pair[1]].update(rd)
                replacement_description = add_and_clean_replacement_pairs(replacement_description)
                if branching_config['replacement_description_saving']:
                    with open(branching_config['replacement_description_saving_location'], 'w', encoding='utf-8') as f:
                        json.dump(replacement_description, f)
            else:
                replacement_description = add_and_clean_replacement_pairs(replacement_description)

            rd = replacement_description[category_pair[0]][category_pair[1]]
            df_new = df[df['category'] == category_pair[0]].copy()
            df_new['prompts'] = df_new['prompts'].apply(lambda x: replace_terms(x, rd).capitalize())
            if branching_config['counterfactual_baseline']:
                df_new['baseline'] = df_new['baseline'].apply(lambda x: replace_terms(x, rd).capitalize())
            df_new['source_tag'] = df_new.apply(lambda row: f'br_{row["source_tag"]}_cat_{row["category"]}', axis=1)
            df_new['category'] = df_new['category'].apply(lambda x: replace_terms(x, rd))
            df_new['keyword'] = df_new['keyword'].apply(lambda x: replace_terms(x, rd))
            df_result = pd.concat([df_result, df_new])

            self.output_df = df_result

        return self.scraped_sentence_saged_data()


