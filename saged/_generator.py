from ._utility import check_benchmark, check_generation_function

class ResponseGenerator:
    def __init__(self, benchmark):
        check_benchmark(benchmark)
        self.benchmark = benchmark

    @classmethod
    def prompt_template(cls, task):
        if task == 'completion':
            return lambda x: f'Compete the following one sentence to make it more meaningful: "{x}"'
        if task == 'question_answering':
            return lambda x: f'Answer the following question in one sentence: "{x}"'

    def generate(self,
                 generation_function,
                 generation_name='LLM',
                 remove_prompt = False,
                 add_prompt_to_generation = False,
                 max_generation_length=1000):

        check_generation_function(generation_function)
        generation = generation_function
        print('Generating.....')
        self.benchmark[generation_name] = self.benchmark['prompts'].progress_apply(generation)
        self.benchmark[generation_name] = self.benchmark.apply(lambda x: x[generation_name][:max_generation_length],
                                                               axis=1)
        if add_prompt_to_generation:
            self.benchmark[generation_name] = self.benchmark.apply(lambda x: x['prompts'] + x[generation_name],
                                                                   axis=1)
        if remove_prompt:
            self.benchmark['baseline'] = self.benchmark.apply(lambda x: x['baseline'].replace(x['prompts'], ''), axis=1)
            self.benchmark[generation_name] = self.benchmark.apply(lambda x: x[generation_name].replace(x['prompts'], ''),
                                                                   axis=1)
        # notice that some model has maximal length requirement
        return self.benchmark

