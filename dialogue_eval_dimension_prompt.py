import re
import copy
import math
import collections
import numpy as np


def _merge_sample_prompt_and_score(fewshot_prompt, sample_prompt):
    fewshot_placeholders = re.findall(r'(\{.*?})', fewshot_prompt)
    fewshot_placeholders = set([x for x in fewshot_placeholders if x != '{sample_prompt}'])
    fewshot_placeholders = sorted(list(fewshot_placeholders))
    for placeholder in fewshot_placeholders:
        fewshot_prompt = fewshot_prompt.replace(placeholder, '{' + placeholder + '}')
    fewshot_prompt = fewshot_prompt.format(sample_prompt=sample_prompt)
    return fewshot_prompt


def _get_min_max_score(score_options, is_normalize_score, normalize_score_max=10):
    score_options = sorted(score_options)
    min_max_score = [score_options[0], score_options[-1]]
    origin_min_max_score = copy.deepcopy(min_max_score)
    if is_normalize_score:
        min_max_score = [math.ceil(normalize_score_max * (x / score_options[-1])) for x in min_max_score]
    min_max_score = tuple(min_max_score)
    origin_min_max_score = tuple(origin_min_max_score)
    return min_max_score, origin_min_max_score


def unmodifiable(func):
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        value = copy.deepcopy(value)
        return value

    return wrapper


class DialogueEvalDimensionPrompt:
    def __init__(self,
                 eval_dim: str,
                 few_shot_k: int,
                 eval_dim_config: dict,
                 default_config: dict,
                 is_normalize_score: bool = False,
                 normalize_score_max: int = 10
                 ):
        eval_dim_config = copy.deepcopy(eval_dim_config)
        default_config = copy.deepcopy(default_config)
        self.eval_dim = eval_dim
        self.few_shot_k = few_shot_k
        self.is_normalize_score = is_normalize_score
        self.normalize_score_max = normalize_score_max

        _to_read_keys = (('sample_prompt', True, True),
                         ('fewshot_prompt', True, True),
                         ('zeroshot_prompt', True, True),
                         ('prompt', True, True),
                         ('end_symbol', True, True),
                         ('description', True, False),
                         ('scale', True, True),
                         ('scale_mapping', False, None),
                         ('score_desc', False, None),
                         ('score_desc_for_float', False, None),
                         ('scale_type', True, True),
                         ('is_choice_to_score', False, None),
                         ('is_add_few_show_separator', False, None),
                         ('few_show_separator', False, None),
                         ('special_contexts', False, None)
                         )
        prompts_dict = {}
        for prompt_key, is_required, use_general_default in _to_read_keys:

            if is_required:
                if use_general_default:
                    prompts_dict[prompt_key] = eval_dim_config.get(prompt_key,
                                                                   default_config[prompt_key])
                else:
                    prompts_dict[prompt_key] = eval_dim_config[prompt_key]
            else:
                prompts_dict[prompt_key] = eval_dim_config.get(prompt_key,
                                                               default_config.get(prompt_key, None))

            if prompt_key == 'scale_type':
                if prompts_dict[prompt_key] == 'choice':
                    assert prompts_dict['scale_mapping'] is not None
            elif prompt_key == 'is_choice_to_score':
                if prompts_dict[prompt_key] is not None:
                    if prompts_dict[prompt_key] is True:
                        assert prompts_dict['score_desc_for_float'] is not None
                    else:
                        assert prompts_dict['score_desc'] is not None

        # post process min max score
        if prompts_dict['scale_type'] in {'int', 'float'}:
            score_options = prompts_dict['scale'].split(',')  # '0,1,2,3' -> ['0','1','2','3']
            score_options = tuple([float(x) for x in score_options])  # ['0','1','2','3'] -> [0.0, 1.0, 2.0, 3.0]
            min_max_score, origin_min_max_score = _get_min_max_score(score_options, is_normalize_score,
                                                                     normalize_score_max=normalize_score_max)
            prompts_dict['min_max_score'] = min_max_score
            prompts_dict['score_options'] = score_options
            prompts_dict['origin_min_max_score'] = origin_min_max_score
        elif prompts_dict['scale_type'] in {'choice'}:
            scale_mapping = prompts_dict['scale_mapping']
            score_options = tuple(sorted(list(set(scale_mapping.keys()))))
            if prompts_dict['is_choice_to_score']:
                score_options = tuple([scale_mapping[x] for x in score_options])
                min_max_score, origin_min_max_score = _get_min_max_score(score_options, is_normalize_score,
                                                                         normalize_score_max=normalize_score_max)
                prompts_dict['min_max_score'] = min_max_score
                prompts_dict['origin_min_max_score'] = origin_min_max_score
            else:
                prompts_dict['min_max_score'] = sorted(list(set(scale_mapping.keys())))
            prompts_dict['score_options'] = score_options
        else:
            raise NotImplementedError

        self._prompts_dict = prompts_dict

    @property
    @unmodifiable
    def prompts_dict(self):
        return self._prompts_dict

    @property
    def scale_range_str(self):
        min_max_score = self._prompts_dict['min_max_score']
        scale_range = f'{min_max_score[0]} to {min_max_score[-1]}'
        return scale_range

    @property
    def scale_type(self):
        return self.prompts_dict['scale_type']

    @property
    def scale_mapping(self):
        return self.prompts_dict['scale_mapping']

    @property
    def min_max_score(self):
        return self.prompts_dict['min_max_score']

    @property
    def score_options(self):
        return self.prompts_dict['score_options']

    @property
    def origin_min_max_score(self):
        return self.prompts_dict['origin_min_max_score']

    @property
    def is_choice_to_score(self):
        return self.prompts_dict['is_choice_to_score']

    def _normalize_score(self, score):
        return round(self.normalize_score_max * (score / self.origin_min_max_score[-1]), 2)

    def fill_one_sample(self, sample_x):
        special_contexts = self.prompts_dict['special_contexts']
        if special_contexts is None:
            sample_x_context = sample_x['context']
        else:
            sample_x_context = ''
            for context_key in special_contexts:
                sample_x_context += sample_x[context_key] + '\n'
        response = sample_x.get('response', '')
        fact = sample_x.get('fact', '')
        sample_prompt = self.prompts_dict['sample_prompt'].format(context=sample_x_context,
                                                                  response=response,
                                                                  fact=fact)
        sample_prompt = sample_prompt.strip()
        return sample_prompt

    def get_score_from_anno_dict(self, anno_dict):
        eval_dim_anno = self.eval_dim
        if eval_dim_anno == 'Uses_Knowledge':
            eval_dim_anno = 'Uses Knowledge'
        elif eval_dim_anno == 'Maintains_Context':
            eval_dim_anno = 'Maintains Context'

        scores = anno_dict[eval_dim_anno]['score']

        # get score
        if self.prompts_dict['scale_type'] in {'int', 'float'}:
            for x in scores: assert x >= 0
            score = np.mean(scores)
            if self.is_normalize_score:
                score = self._normalize_score(score)
            score = round(score, 2)
            assert self.min_max_score[0] <= score <= self.min_max_score[-1]
        elif self.prompts_dict['scale_type'] in {'choice'}:
            if self.prompts_dict['is_choice_to_score']:
                scores = [self.prompts_dict['scale_mapping'][x] for x in scores]
                for x in scores: assert x >= 0
                score = np.mean(scores)
                if self.is_normalize_score:
                    score = self._normalize_score(score)
                score = round(score, 2)
                assert self.min_max_score[0] <= score <= self.min_max_score[-1]
            else:
                score = list(collections.Counter(scores).items())[0][0]
        else:
            raise NotImplementedError
        return score

    def fill_sample_prompt_with_score(self, sample_x):
        mean_score = self.get_score_from_anno_dict(sample_x['annotations'])
        sample_prompt_filled = self.fill_one_sample(sample_x)

        fewshot_prompt = self.prompts_dict['fewshot_prompt']

        if self.is_choice_to_score:
            score_desc = self.prompts_dict['score_desc_for_float']
        else:
            score_desc = self.prompts_dict['score_desc']

        fewshot_prompt = fewshot_prompt.format(
            sample_prompt=sample_prompt_filled,
            aspect=self.eval_dim,
            score=mean_score,
            end_symbol=self.prompts_dict['end_symbol'],
            score_desc=score_desc,
            aspect_description=self.prompts_dict['description']
        )
        if self.prompts_dict['is_add_few_show_separator']:
            fewshot_prompt = f'\n{self.prompts_dict["few_show_separator"] * 78}\n' + fewshot_prompt
        else:
            fewshot_prompt = '\n\n' + fewshot_prompt

        return fewshot_prompt

    def fill_few_shot_demonstrations(self, few_shot_xs):
        if len(few_shot_xs) > 0:
            few_shot_demo_str = ''
            for sample_i, sample_x in enumerate(few_shot_xs):
                sample_demo = self.fill_sample_prompt_with_score(sample_x)
                few_shot_demo_str += f'{sample_demo}'
            few_shot_demo_str = few_shot_demo_str.strip()
        else:
            few_shot_demo_str = []
            for score in self.score_options:
                if self.scale_type in {'int', 'float'}:
                    if self.is_normalize_score:
                        score = self._normalize_score(score)
                elif self.scale_type == 'choice':
                    if self.prompts_dict['is_choice_to_score']:
                        if self.is_normalize_score:
                            score = self._normalize_score(score)
                else:
                    raise NotImplementedError

                if self.is_choice_to_score:
                    score_desc = self.prompts_dict['score_desc_for_float']
                else:
                    score_desc = self.prompts_dict['score_desc']

                score_demo = self.prompts_dict['zeroshot_prompt'].format(score=score,
                                                                         aspect=self.eval_dim,
                                                                         score_desc=score_desc,
                                                                         end_symbol=self.prompts_dict['end_symbol'])
                few_shot_demo_str.append(score_demo)
            few_shot_demo_str = '\n'.join(few_shot_demo_str)

        return few_shot_demo_str

    def fill_dimension_prompt(self, few_shot_demo_str, infer_sample_str):

        if self.is_choice_to_score:
            score_desc = self.prompts_dict['score_desc_for_float']
        else:
            score_desc = self.prompts_dict['score_desc']

        if self.prompts_dict['is_add_few_show_separator']:
            infer_sample_str = f'\n{self.prompts_dict["few_show_separator"] * 78}\n' + infer_sample_str
        else:
            infer_sample_str = '\n' + infer_sample_str

        few_shot_demo_str = few_shot_demo_str + f"\n{'-' * 78}"

        return self.prompts_dict['prompt'].format(fewshot_prompt=few_shot_demo_str,
                                                  aspect=self.eval_dim,
                                                  scale=self.scale_range_str,
                                                  sample_prompt=infer_sample_str,
                                                  aspect_description=self.prompts_dict['description'],
                                                  score_desc=score_desc)
