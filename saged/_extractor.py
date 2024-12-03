import pandas as pd

from ._utility import ignore_future_warnings
from tqdm import tqdm
from transformers import pipeline
import warnings
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity
from ._utility import check_benchmark, clean_sentences_and_join

tqdm.pandas()


class FeatureExtractor:
    def __init__(self, benchmark, generations=('baseline', 'LLM'), calibration = False, baseline ='baseline', embedding_model = None):
        check_benchmark(benchmark)
        for col in generations:
            assert col in benchmark.columns, f"Column '{col}' not found in benchmark"
        self.benchmark = benchmark
        self.generations = [generations] if isinstance(generations, str) else generations
        self.calibration = calibration
        self.baseline = baseline
        self.classification_features = []
        self.cluster_features = []
        self.calibrated_features = []
        self._model = False if embedding_model is None else True
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') if embedding_model is None else embedding_model


    @staticmethod
    def _relabel(text, pipe):
        regard_results = {}
        for _dict in pipe(text, top_k=20):
            regard_results[_dict['label']] = _dict['score']
        return regard_results

    def _baseline_calibration(self, features):
        baseline = self.baseline
        df = self.benchmark
        for feature in features:
            for col in self.generations:
                # if col != baseline:
                df[f'{col}_{feature}_cbr_{baseline}'] = df.apply(lambda x: x[f'{col}_{feature}'] - x[f'{baseline}_{feature}'], axis=1)
            self.calibrated_features.append(f'{feature}_cbr_{baseline}')
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def sentiment_classification(self):
        df = self.benchmark
        print('Using default sentiment classifier: lxyuan/distilbert-base-multilingual-cased-sentiments-student')
        warnings.filterwarnings("ignore", category=FutureWarning)
        sentiment_classifier = pipeline("text-classification",
                                        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
        warnings.filterwarnings("default", category=FutureWarning)

        for col in self.generations:
            df[f'{col}_sentiment_temp'] = df[col].progress_apply(lambda text: FeatureExtractor._relabel(text, sentiment_classifier))
            df[f'{col}_sentiment_score'] = df[f'{col}_sentiment_temp'].apply(
                lambda x: (x['positive'] - x['negative'] + 1)/2)
            df.drop(columns=[f'{col}_sentiment_temp'], inplace=True)

        self.classification_features.append('sentiment_score')
        if self.calibration:
            df = self._baseline_calibration(['sentiment_score'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def regard_classification(self):
        df = self.benchmark
        print('Using default regard classifier: sasha/regardv3')
        regard_classifier = pipeline("text-classification", model="sasha/regardv3")

        for col in self.generations:
            df[f'{col}_regard_temp'] = df[col].progress_apply(lambda text: FeatureExtractor._relabel(text, regard_classifier))
            df[f'{col}_regard_score'] = df[f'{col}_regard_temp'].apply(
                lambda x: x['positive'] - x['negative'] + 1)
            df.drop(columns=[f'{col}_regard_temp'], inplace=True)

        self.classification_features.append('regard_score')
        if self.calibration:
            df = self._baseline_calibration(['regard_score'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def stereotype_classification(self):
        df = self.benchmark
        print('Using default stereotype classifier: holistic-ai/stereotype-deberta-v3-base-tasksource-nli')
        stereotype_classifier = pipeline("text-classification", model="holistic-ai/stereotype-deberta-v3-base-tasksource-nli")

        for col in self.generations:
            df[f'{col}_temp'] = df[col].progress_apply(lambda text: FeatureExtractor._relabel(text, stereotype_classifier))
            df[f'{col}_stereotype_gender_score'] = df[f'{col}_temp'].apply(
                lambda x: x['stereotype_gender'])
            df[f'{col}_stereotype_religion_score'] = df[f'{col}_temp'].apply(
                lambda x: x['stereotype_religion'])
            df[f'{col}_stereotype_profession_score'] = df[f'{col}_temp'].apply(
                lambda x: x['stereotype_profession'])
            df[f'{col}_stereotype_race_score'] = df[f'{col}_temp'].apply(
                lambda x: x['stereotype_race'])
            df.drop(columns=[f'{col}_temp'], inplace=True)

        self.classification_features.extend(['stereotype_gender_score', 'stereotype_religion_score', 'stereotype_profession_score', 'stereotype_race_score'])
        if self.calibration:
            df = self._baseline_calibration(['stereotype_gender_score', 'stereotype_religion_score', 'stereotype_profession_score', 'stereotype_race_score'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def personality_classification(self):
        df = self.benchmark
        print('Using default personality classifier: Navya1602/editpersonality_classifier')
        stereotype_classifier = pipeline("text-classification", model="Navya1602/editpersonality_classifier")

        for col in self.generations:
            df[f'{col}_temp'] = df[col].progress_apply(lambda text: FeatureExtractor._relabel(text, stereotype_classifier))
            df[f'{col}_extraversion_score'] = df[f'{col}_temp'].apply(
                lambda x: x['extraversion'])
            df[f'{col}_neuroticism_score'] = df[f'{col}_temp'].apply(
                lambda x: x['neuroticism'])
            df[f'{col}_agreeableness_score'] = df[f'{col}_temp'].apply(
                lambda x: x['agreeableness'])
            df[f'{col}_conscientiousness_score'] = df[f'{col}_temp'].apply(
                lambda x: x['conscientiousness'])
            df[f'{col}_openness_score'] = df[f'{col}_temp'].apply(
                lambda x: x['openness'])
            df.drop(columns=[f'{col}_temp'], inplace=True)

        self.classification_features.extend(['extraversion_score', 'neuroticism_score', 'agreeableness_score', 'conscientiousness_score', 'openness_score'])
        if self.calibration:
            df = self._baseline_calibration(['extraversion_score', 'neuroticism_score', 'agreeableness_score', 'conscientiousness_score', 'openness_score'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def toxicity_classification(self, model_choice='detoxify'):
        df = self.benchmark
        if model_choice == 'detoxify':
            print('Using unitary/toxic-bert model for toxicity classification')
            toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
        elif model_choice == 'junglelee':
            print('Using JungleLee/bert-toxic-comment-classification for toxicity classification')
            toxicity_classifier = pipeline("text-classification", model="JungleLee/bert-toxic-comment-classification")
        else:
            raise ValueError("Invalid model_choice. Choose either 'detoxify' or 'junglelee'.")

        def get_toxicity_score(text):
            result = FeatureExtractor._relabel(text, toxicity_classifier)
            return result['toxic']

        for col in self.generations:
            df[f'{col}_toxicity_score'] = df[col].progress_apply(get_toxicity_score)
        self.classification_features.append('toxicity_score')
        if self.calibration:
            df = self._baseline_calibration(['toxicity_score'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def customized_classification(self, classifier_name, classifier):
        df = self.benchmark

        for col in self.generations:
            df[f'{col}_{classifier_name}_score'] = df[col].progress_apply(classifier)


        self.classification_features.append(f'{classifier_name}_score')
        if self.calibration:
            df = self._baseline_calibration([f'{classifier_name}_score'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def embedding_distance(self, distance_function = 'cosine', custom_distance_fn = None):
        if not self._model:
            print('Using default embedding model: all-MiniLM-L6-v2')
        def calculate_pairwise_distances(generated_answers, expected_answers,
                                         distance_function=distance_function, custom_distance_fn=custom_distance_fn):
            """
            Calculate a pairwise distance for a set of generated and expected answers using the specified distance function.

            Parameters:
            - generated_answers (list of str): A list of answers generated by the model.
            - expected_answers (list of str): A list of corresponding expected answers.
            - distance_function (str): The type of distance function to use ('cosine', 'l1', 'l2', 'custom').
            - custom_distance_fn (callable): A custom distance function provided by the user. Should accept two arrays of vectors.
            - model: The embedding model to use for generating embeddings. If None, self.embedding_model is used.

            Returns:
            - distances (pd.DataFrame): A DataFrame with pairwise distances between each generated and expected answer.
            """
            if len(generated_answers) != len(expected_answers):
                raise ValueError("The number of generated answers and expected answers must be the same.")

            model = self.embedding_model  # Assumes self.embedding_model is defined elsewhere

            if distance_function == 'custom' and not callable(custom_distance_fn):
                raise ValueError(
                    "custom_distance_fn must be provided and callable when using 'custom' distance function.")

            # Generate embeddings for generated and expected answers
            generated_vectors = model.encode(generated_answers, convert_to_numpy=True)
            expected_vectors = model.encode(expected_answers, convert_to_numpy=True)

            # Initialize an empty list to store distances
            distances = []

            if distance_function == 'cosine':
                def cosine_similarity(v1, v2):
                    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

                for e1, e2 in zip(generated_vectors, expected_vectors):
                    cosine_distance = 1 - cosine_similarity(e1, e2)
                    distances.append(cosine_distance)

            elif distance_function == 'l1':
                for e1, e2 in zip(generated_vectors, expected_vectors):
                    l1_distance = np.sum(np.abs(e1 - e2))
                    distances.append(l1_distance)

            elif distance_function == 'l2':
                for e1, e2 in zip(generated_vectors, expected_vectors):
                    l2_distance = np.linalg.norm(e1 - e2)
                    distances.append(l2_distance)

            elif distance_function == 'custom':
                for e1, e2 in zip(generated_vectors, expected_vectors):
                    custom_distance = custom_distance_fn(e1, e2)
                    distances.append(custom_distance)

            else:
                raise ValueError("Unsupported distance function. Choose from 'cosine', 'l1', 'l2', or 'custom'.")

            return distances

        df = self.benchmark
        for col in self.generations:
            df[f'{col}_{distance_function}_distance_wrt_{self.baseline}'] = calculate_pairwise_distances(df[col], df[self.baseline])
        self.classification_features.append(f'{distance_function}_distance_wrt_{self.baseline}')
        self.benchmark = df.copy()
        if self.calibration:
            df = self._baseline_calibration([f'{distance_function}_distance_wrt_{self.baseline}'])
        self.benchmark = df.copy()
        return df

    @ignore_future_warnings
    def cluster_and_label(self, top_word_insight=True, num_clusters=3, segregation=None, anchored = False, unpivot = False):
        if not self._model:
            print('Using default embedding model: all-MiniLM-L6-v2')
        generations = self.generations if not anchored else [self.baseline]

        def _extract_embeddings(sentences):
            model = self.embedding_model
            embeddings = []
            for sentence in tqdm(sentences, desc="Encoding sentences"):
                embeddings.append(model.encode(sentence))
            return np.array(embeddings)

        def _find_most_similar_words(sentences, word_list=None, given_word=None, method='average',
                                        filter_similar_words=True):

            """
            Find the most similar words in a list to a group of sentences, with an option to filter out words
            that are 0.5 or more similar to a given word.

            Parameters:
            sentences (list of str): The group of sentences to compare.
            word_list (list of str): The list of words to rank by similarity. Defaults to Oxford 5000 if not provided.
            given_word (str): The word to which similarity should be checked for filtering.
            method (str): The method to combine sentence embeddings. Options are 'average', 'max_pooling',
                          'weighted_average', 'concatenation'. Default is 'average'.
            model_name (str): The name of the SentenceTransformer model to use. Default is 'paraphrase-Mpnet-base-v2'.
            filter_similar_words (bool): Whether to filter out words that are 0.5 or more similar to the given word. Default is True.

            Returns:
            list of tuple: A list of tuples where each tuple contains a word and its similarity score, sorted by similarity.
            """

            def load_oxford5000(file_path='Oxford 5000.txt'):
                """Load the Oxford 5000 word list from a file."""
                with open(file_path, 'r') as file:
                    oxford5000_words = [line.strip() for line in file]
                return oxford5000_words

            # Step 1: Load the Oxford 5000 word list if word_list is not provided
            if word_list is None:
                print("Loading Oxford 5000 word list, because word_list is not given...")
                word_list = load_oxford5000()

            # Step 2: Load the SentenceTransformer model
            model = self.embedding_model

            # Step 3: Get embeddings for sentences
            sentence_embeddings = model.encode(sentences)

            # Step 4: Get embedding for the given word
            given_word_vector = None
            if given_word is not None:
                given_word_vector = model.encode([given_word])[0]

            # Step 5: Combine sentence embeddings using the chosen method
            if method == 'average':
                combined_sentence_vector = np.mean(sentence_embeddings, axis=0)
            elif method == 'max_pooling':
                combined_sentence_vector = np.max(sentence_embeddings, axis=0)
            elif method == 'weighted_average' and given_word_vector is not None:
                # Calculate similarities of each sentence to the given word
                similarities = cosine_similarity(sentence_embeddings, [given_word_vector]).flatten()

                # Calculate weights as the inverse of similarities (higher weight for less similar sentences)
                weights = 1 - similarities

                # Avoid division by zero by adding a small epsilon value
                epsilon = 1e-6
                combined_sentence_vector = np.sum(weights[:, None] * sentence_embeddings, axis=0) / (
                            np.sum(weights) + epsilon)
            elif method == 'concatenation':
                concatenated_sentence = " ".join(sentences)
                combined_sentence_vector = model.encode([concatenated_sentence])[0]
            else:
                raise ValueError(
                    "Invalid method specified. Choose from 'average', 'max_pooling', 'weighted_average', 'concatenation'.")

            # Step 6: Get embeddings for the word list
            word_vectors = model.encode(word_list)

            # Step 7: Optionally filter out words that are 0.5 or more similar to the given word
            if filter_similar_words and given_word:
                filtered_words = []
                filtered_vectors = []
                for word, vector in zip(word_list, word_vectors):
                    similarity = cosine_similarity([given_word_vector], [vector]).flatten()[0]
                    if similarity < 0.5:
                        filtered_words.append(word)
                        filtered_vectors.append(vector)
            else:
                filtered_words = word_list
                filtered_vectors = word_vectors

            # Step 8: Compute cosine similarity between the combined sentence vector and each filtered word vector
            similarities = cosine_similarity([combined_sentence_vector], filtered_vectors).flatten()

            # Step 9: Rank words by similarity
            ranked_words = sorted(zip(filtered_words, similarities), key=lambda x: x[1], reverse=True)

            return ranked_words

        def _find_different_top_words(word_dicts, top_n=4, similarity_threshold=0.9):
            # Load the model
            model = self.embedding_model

            # Step 1: Initialize an empty result dictionary
            result = {key: [] for key in word_dicts.keys()}
            buffer_n = top_n + 10

            # Step 2: Precompute embeddings for all words
            embeddings_dict = {key: model.encode(words) for key, words in word_dicts.items()}

            # Step 3: Iterate through the lists, adding words and checking conditions
            index = 0
            while any(len(lst) < buffer_n for lst in result.values()):
                for key in word_dicts:
                    if len(result[key]) < buffer_n and index < len(word_dicts[key]):
                        word_to_add = word_dicts[key][index]
                        new_embedding = embeddings_dict[key][index].reshape(1, -1)

                        # Check similarity with each existing word in the result list
                        add_word = True
                        for existing_word in result[key]:
                            existing_embedding = model.encode([existing_word]).reshape(1, -1)
                            similarity = cosine_similarity(new_embedding, existing_embedding)[0][0]
                            if similarity > similarity_threshold:
                                add_word = False
                                break

                        if add_word:
                            result[key].append(word_to_add)

                index += 1
                if index >= len(list(word_dicts.values())[0]):
                    break

            # Step 4: Trim the results to only the top_n words
            for key in result.keys():
                if len(result[key]) > top_n:
                    result[key] = result[key][:top_n]
            return result

        def _generate_cluster_themes(df, cluster_text_col_dict, top_n=3):
            cluster_themes = {}

            # Concatenate all cluster columns to find unique clusters
            all_clusters = pd.concat([df[cluster_col] for cluster_col in cluster_text_col_dict.keys()])
            unique_clusters = all_clusters.unique()

            for cluster in unique_clusters:
                cluster_texts = []

                # Concatenate all the texts from the specified text columns where the cluster matches
                for cluster_col, text_col in cluster_text_col_dict.items():
                    cluster_texts.extend(df[df[cluster_col] == cluster][text_col].tolist())

                # Clean and join the texts
                cleaned_texts = clean_sentences_and_join(cluster_texts)

                # Find similar words (assumes a function to do this is defined elsewhere)
                similar_words = _find_most_similar_words(
                    cluster_texts,
                    word_list=list(set(cleaned_texts.split())),
                    given_word=cluster.split('_')[0],
                    method='concatenation',
                    filter_similar_words=True
                )

                # Store the similar words as themes
                cluster_themes[cluster] = [word for word, _ in similar_words]

            # Filter and select top words based on similarity
            cluster_themes = _find_different_top_words(cluster_themes.copy(), top_n=top_n, similarity_threshold=0.5)

            return cluster_themes

        def _pivot_clustered_df(clustered_df, segregation, targets, segs, cluster_themes=None, top_word_insight=False):
            pivoted_df = pd.DataFrame(index=clustered_df.index)

            for seg in segs:
                for generation in targets:
                    cluster_names = clustered_df.loc[clustered_df[segregation] == seg, f'{generation}_cluster'].unique()
                    for cluster_name in cluster_names:
                        if top_word_insight:
                            theme = cluster_themes.get(cluster_name, [])
                            theme_str = '_'.join(theme) if theme else cluster_name
                            cluster_col_name = f'{theme_str}'
                        else:
                            cluster_col_name = f'{cluster_name}'

                        pivoted_df[f'{generation}_cluster_{cluster_col_name}'] = np.where(
                            (clustered_df[segregation] == seg) & (
                                        clustered_df[f'{generation}_cluster'] == cluster_name), 1,
                            np.where(clustered_df[segregation] == seg, 0, np.nan)
                        )
                        self.cluster_features.append(f'cluster_{cluster_col_name}')
            return pivoted_df


        df = self.benchmark.copy()
        clustered_df = self.benchmark.copy()

        assert segregation is None or segregation in ['category', 'domain', 'source_tag'], "segregation must be None, 'category',  'source_tag', or 'domain'."

        segs = df[segregation].unique() if segregation else ['ALL']

        for seg in segs:
            if segregation:
                seg_df = df[df[segregation] == seg]
            else:
                seg_df = df
            for generation in generations:
                sentences = seg_df[generation].tolist()
                embeddings = _extract_embeddings(sentences)

                if embeddings.size == 0:
                    print(f"Warning: No embeddings generated for {seg} in {generation}. Skipping clustering.")
                    clustered_df.loc[seg_df.index, f'{generation}_cluster'] = np.nan
                    continue

                n_clusters = min(num_clusters, len(embeddings))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                clusters = kmeans.fit_predict(embeddings)

                # Convert cluster labels to strings and prefix with category name
                cluster_labels = [f"{seg}_{str(label)}" for label in clusters]

                # Assign cluster labels to the DataFrame
                clustered_df.loc[seg_df.index, f'{generation}_cluster'] = cluster_labels

        if top_word_insight:
            self.cluster_themes = {}
            cluster_text_col_dict = {f'{generation}_cluster': generation for generation in generations}
            self.cluster_themes = _generate_cluster_themes(clustered_df, cluster_text_col_dict, top_n=3)
            for generation in generations:
                clustered_df[f'{generation}_cluster_theme'] = clustered_df[f'{generation}_cluster'].apply(
                    lambda x: ', '.join(self.cluster_themes.get(x, [])))

        self.benchmark = clustered_df.copy()

        if anchored or unpivot:
            return clustered_df

        # Pivot the table using the helper function
        pivoted_df = _pivot_clustered_df(clustered_df, segregation, generations, segs, self.cluster_themes, top_word_insight)
        # Combine the original columns with the pivoted columns
        combined_df = pd.concat([clustered_df, pivoted_df], axis=1)

        self.benchmark = combined_df.copy()

        return combined_df

    @ignore_future_warnings
    def cluster_and_sort(self, top_word_insight=True, **kwargs):
        if not self._model:
            print('Using default embedding model: all-MiniLM-L6-v2')

        def _cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        def _compute_combined_sentence_vector(anchor_sentences, method='concatenation', given_word=None):
            # Load the SentenceTransformer model
            model = self.embedding_model

            # Get embeddings for anchor sentences
            sentence_embeddings = model.encode(anchor_sentences)

            # Get embedding for the given word
            given_word_vector = None
            if given_word is not None:
                given_word_vector = model.encode([given_word])[0]

            # Combine sentence embeddings using the chosen method
            if method == 'average':
                combined_sentence_vector = np.mean(sentence_embeddings, axis=0)
            elif method == 'max_pooling':
                combined_sentence_vector = np.max(sentence_embeddings, axis=0)
            elif method == 'weighted_average' and given_word_vector is not None:
                # Calculate similarities of each sentence to the given word
                similarities = np.array(
                    [_cosine_similarity(sentence, given_word_vector) for sentence in sentence_embeddings])

                # Calculate weights as the inverse of similarities (higher weight for less similar sentences)
                weights = 1 - similarities

                # Avoid division by zero by adding a small epsilon value
                epsilon = 1e-6
                combined_sentence_vector = np.sum(weights[:, None] * sentence_embeddings, axis=0) / (
                            np.sum(weights) + epsilon)
            elif method == 'concatenation':
                concatenated_sentence = " ".join(anchor_sentences)
                combined_sentence_vector = model.encode([concatenated_sentence])[0]
            else:
                raise ValueError(
                    "Invalid method specified. Choose from 'average', 'max_pooling', 'weighted_average', 'concatenation'.")

            return combined_sentence_vector

        def _compute_distance_with_vector(st_to_sort, combined_sentence_vector, distance_function='cosine'):
            # Get embedding for the sentence to sort
            st_to_sort_vector = self.embedding_model.encode([st_to_sort])[0]

            # Compute the distance between st_to_sort and the combined sentence vector
            if distance_function == 'cosine':
                cosine_dist = 1 - _cosine_similarity(st_to_sort_vector, combined_sentence_vector)
                return cosine_dist
            elif distance_function == 'l1':
                l1_dist = np.sum(np.abs(st_to_sort_vector - combined_sentence_vector))
                return l1_dist
            elif distance_function == 'l2':
                l2_dist = np.linalg.norm(st_to_sort_vector - combined_sentence_vector)
                return l2_dist
            else:
                raise ValueError("Invalid distance function specified. Choose from 'cosine', 'l1', 'l2'.")

        # Function to find the name of the anchor sentence group with the minimum distance
        def _find_closest_anchor(st_to_sort, combined_vectors):
            min_distance = float('inf')
            closest_anchor_name = None

            for name, combined_vector in combined_vectors.items():
                distance = _compute_distance_with_vector(st_to_sort, combined_vector, distance_function='cosine')
                if distance < min_distance:
                    min_distance = distance
                    closest_anchor_name = name

            return closest_anchor_name

        def _pivot_clustered_df(clustered_df, segregation, targets, segs, cluster_themes=None, top_word_insight=False, baseline=None):
            pivoted_df = pd.DataFrame(index=clustered_df.index)

            for seg in segs:
                for generation in targets:
                    cluster_names = clustered_df.loc[clustered_df[segregation] == seg, f'{generation}_cluster_st_{baseline}'].unique()
                    for cluster_name in cluster_names:
                        if top_word_insight:
                            theme = cluster_themes.get(cluster_name, [])
                            theme_str = '_'.join(theme) if theme else cluster_name
                            cluster_col_name = f'{theme_str}'
                        else:
                            cluster_col_name = f'{cluster_name}'

                        pivoted_df[f'{generation}_cluster_{cluster_col_name}_st_{baseline}'] = np.where(
                            (clustered_df[segregation] == seg) & (
                                        clustered_df[f'{generation}_cluster_st_{baseline}'] == cluster_name), 1,
                            np.where(clustered_df[segregation] == seg, 0, np.nan)
                        )

                        self.cluster_features.append(f'cluster_{cluster_col_name}_st_{baseline}')

            return pivoted_df

        # Apply the filter based on the segregation value of each row in df_to_sort
        def _filter_and_find_closest(row, segregation):
            current_segregation = row[segregation]
            # Filter the combined_vectors_of_anchor dictionary based on current_segregation
            filtered_vectors = {
                name: vector
                for (name, seg_value), vector in combined_vectors_of_anchor.items()
                if seg_value == current_segregation
            }
            return _find_closest_anchor(row[target], combined_vectors=filtered_vectors)

        baseline = self.baseline
        df_anchor = self.cluster_and_label(anchored=True, **kwargs).copy()
        segregation = kwargs.get('segregation', None)

        if top_word_insight:
            baseline_cluster_column_name = f'{baseline}_cluster_theme'
        else:
            baseline_cluster_column_name = f'{baseline}_cluster'

        grouped_anchor_sentences = df_anchor.groupby([baseline_cluster_column_name, segregation])[baseline].apply(
            list)

        # Make method optional
        if 'method' in kwargs and kwargs['method']:
            method = kwargs['method']
        else:
            method = 'concatenation'

        combined_vectors_of_anchor = {
            (name, segregation_value): _compute_combined_sentence_vector(sentences, method=method)
            for (name, segregation_value), sentences in grouped_anchor_sentences.items()
        }

        df_to_sort = self.benchmark.copy()

        for target in self.generations:
            if target != baseline:
                print(f'Sorting for {target}...')

                df_to_sort[f'{target}_cluster_st_{baseline}'] = df_to_sort.progress_apply(_filter_and_find_closest,
                                                                                          segregation=segregation,
                                                                                          axis=1)

        df_to_sort.rename(columns={f'{baseline}_cluster': f'{baseline}_cluster_st_{baseline}'}, inplace=True)
        self.benchmark = df_to_sort.copy()


        if 'unpivot' in kwargs and kwargs['unpivot']:
            return df_to_sort

        # Apply pivoting to df_to_sort to create binary columns
        segregation = kwargs.get('segregation', None)
        segs = df_to_sort[segregation].unique() if segregation else ['ALL']
        pivoted_df = _pivot_clustered_df(df_to_sort, segregation, self.generations, segs, self.cluster_themes,
                                        kwargs.get('top_word_insight', False), baseline = baseline)

        # Combine the original columns with the pivoted columns
        combined_df = pd.concat([df_to_sort, pivoted_df], axis=1)

        self.benchmark = combined_df.copy()
        return combined_df

    @ignore_future_warnings
    def cluster_and_distance(self, sorting = False, top_word_insight=True, **kwargs):
        if not self._model:
            print('Using default embedding model: all-MiniLM-L6-v2')

        def _cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        def _compute_combined_sentence_vector(anchor_sentences, method='concatenation', given_word=None):
            # Load the SentenceTransformer model
            model = self.embedding_model
            # Get embeddings for anchor sentences
            sentence_embeddings = model.encode(anchor_sentences)

            # Get embedding for the given word
            given_word_vector = None
            if given_word is not None:
                given_word_vector = model.encode([given_word])[0]

            # Combine sentence embeddings using the chosen method
            if method == 'average':
                combined_sentence_vector = np.mean(sentence_embeddings, axis=0)
            elif method == 'max_pooling':
                combined_sentence_vector = np.max(sentence_embeddings, axis=0)
            elif method == 'weighted_average' and given_word_vector is not None:
                # Calculate similarities of each sentence to the given word
                similarities = np.array(
                    [_cosine_similarity(sentence, given_word_vector) for sentence in sentence_embeddings])

                # Calculate weights as the inverse of similarities (higher weight for less similar sentences)
                weights = 1 - similarities

                # Avoid division by zero by adding a small epsilon value
                epsilon = 1e-6
                combined_sentence_vector = np.sum(weights[:, None] * sentence_embeddings, axis=0) / (
                        np.sum(weights) + epsilon)
            elif method == 'concatenation':
                concatenated_sentence = " ".join(anchor_sentences)
                combined_sentence_vector = model.encode([concatenated_sentence])[0]
            else:
                raise ValueError(
                    "Invalid method specified. Choose from 'average', 'max_pooling', 'weighted_average', 'concatenation'.")

            return combined_sentence_vector

        def _compute_distance_with_vector(st_to_sort, combined_sentence_vector, distance_function='cosine'):
            # Get embedding for the sentence to sort
            st_to_sort_vector = self.embedding_model.encode([st_to_sort])[0]

            # Compute the distance between st_to_sort and the combined sentence vector
            if distance_function == 'cosine':
                cosine_dist = 1 - _cosine_similarity(st_to_sort_vector, combined_sentence_vector)
                return cosine_dist
            elif distance_function == 'l1':
                l1_dist = np.sum(np.abs(st_to_sort_vector - combined_sentence_vector))
                return l1_dist
            elif distance_function == 'l2':
                l2_dist = np.linalg.norm(st_to_sort_vector - combined_sentence_vector)
                return l2_dist
            else:
                raise ValueError("Invalid distance function specified. Choose from 'cosine', 'l1', 'l2'.")

        def _create_segregation_cluster_dict(df_anchor, segregation_column, generations, top_word_insight = top_word_insight):
            if segregation_column is None:
                raise ValueError("The segregation_column must not be None. Please provide a valid column name.")

            segregation_cluster_dict = {}

            # Get unique values from the segregation column
            segregation_values = df_anchor[segregation_column].unique()

            for segregation_value in segregation_values:
                # Initialize an empty set to collect unique clusters for this segregation value
                cluster_set = set()

                # Filter the DataFrame by the current segregation value
                df_group = df_anchor[df_anchor[segregation_column] == segregation_value]

                for generation in generations:
                    if top_word_insight:
                        cluster_column_name = f'{generation}_cluster_theme'
                    else:
                        cluster_column_name = f'{generation}_cluster'

                    # Add the unique clusters for this generation to the set
                    cluster_set.update(df_group[cluster_column_name].unique())

                # Convert the set to a list and store it in the dictionary
                segregation_cluster_dict[segregation_value] = list(cluster_set)

            return segregation_cluster_dict

        distance_segregation = kwargs.pop('distance_segregation', None)

        df_anchor = self.cluster_and_label(unpivot = True, top_word_insight = top_word_insight, **kwargs).copy()

        # Dictionary to store the combined vector for each unique cluster
        all_combined_vectors = {}

        generations = self.generations

        segregation_column = kwargs.get('segregation', None)

        if 'method' in kwargs and kwargs['method']:
            method = kwargs['method']
        else:
            method = 'concatenation'

        if (not distance_segregation) or (segregation_column is None):
            unique_clusters = set()  # To store unique cluster values across all generations

            # Step 1: Identify all unique clusters across generations and compute combined vectors
            for generation in generations:
                if top_word_insight:
                    cluster_column_name = f'{generation}_cluster_theme'
                else:
                    cluster_column_name = f'{generation}_cluster'

                unique_clusters.update(df_anchor[cluster_column_name].unique())

            print(f'Unique clusters: {unique_clusters}')
            for cluster in unique_clusters:
                print(f'Computing combined vector for cluster {cluster}...')
                cluster_sentences = []

                for generation in generations:
                    if top_word_insight:
                        cluster_column_name = f'{generation}_cluster_theme'
                    else:
                        cluster_column_name = f'{generation}_cluster'

                    # Collect all sentences that belong to the current cluster
                    cluster_sentences.extend(
                        df_anchor[df_anchor[cluster_column_name] == cluster][generation].dropna().tolist())

                # Compute the combined vector for the current cluster
                all_combined_vectors[cluster] = _compute_combined_sentence_vector(cluster_sentences, method=method)

            # Step 2: Compute distances for each sentence to all clusters and add them to the DataFrame
            distance_df = df_anchor.copy()

            for generation in generations:
                for cluster in unique_clusters:
                    # Create a column name for the distance to each cluster
                    distance_column_name = f'{generation}_{cluster}_distance'

                    # Calculate the distance for each sentence in this generation with respect to the current cluster's combined vector
                    distance_df[distance_column_name] = distance_df.apply(
                        lambda row: _compute_distance_with_vector(
                            row[generation],
                            all_combined_vectors[cluster],
                            distance_function=kwargs.get('distance_function', 'cosine')
                        ),
                        axis=1
                    )

                    self.cluster_features.append(f'{cluster}_distance')

            return distance_df
        else:
            cluster_dict = _create_segregation_cluster_dict(df_anchor, segregation_column, generations,
                                               top_word_insight=top_word_insight)
            # Step 1: Compute combined vectors for each cluster within each segregation group
            for segregation_value, clusters in cluster_dict.items():
                # Filter by segregation group
                df_group = df_anchor[df_anchor[segregation_column] == segregation_value]

                for cluster in clusters:
                    cluster_sentences = []

                    for generation in generations:
                        if top_word_insight:
                            cluster_column_name = f'{generation}_cluster_theme'
                        else:
                            cluster_column_name = f'{generation}_cluster'

                        # Collect all sentences that belong to the current cluster
                        cluster_sentences.extend(
                            df_group[df_group[cluster_column_name] == cluster][generation].dropna().tolist())

                    # Compute the combined vector for the current cluster
                    all_combined_vectors[(segregation_value, cluster)] = _compute_combined_sentence_vector(
                        cluster_sentences, method=method)

            # Step 2: Compute distances for each sentence to all clusters and add them to the DataFrame
            distance_df = df_anchor.copy()

            for generation in generations:
                for segregation_value, clusters in cluster_dict.items():
                    # Filter by segregation group
                    df_group = df_anchor[df_anchor[segregation_column] == segregation_value]

                    for cluster in clusters:
                        # Create a column name for the distance to each cluster
                        distance_column_name = f'{generation}_{segregation_value}_{cluster}_distance'

                        # Calculate the distance for each sentence in this generation with respect to the current cluster's combined vector
                        distance_df[distance_column_name] = df_group.apply(
                            lambda row: _compute_distance_with_vector(
                                row[generation],
                                all_combined_vectors[(segregation_value, cluster)],
                                distance_function=kwargs.get('distance_function', 'cosine')
                            ) if (segregation_value, cluster) in all_combined_vectors else np.nan,
                            axis=1
                        )

                        self.cluster_features.append(f'{segregation_value}_{cluster}_distance')
            return distance_df
