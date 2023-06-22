import numpy as np
import re
import os
import sys
import json
import hashlib
import datetime

def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding=encoding) as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def get_llm_response_cache_path(chatgpt_request_dict, prompt, cache_dir):
    cache_md5 = hashlib.md5((str(chatgpt_request_dict) + prompt).encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f'{cache_md5}.json')
    return cache_path

def score_by_llm(eval_dim_prompts,
                 text_gen_api,
                 model_name,
                 temperature,
                 top_p_value,
                 max_new_tokens,
                 max_workers,
                 end_symbol,
                 verbose,
                 min_max_score,
                 scale_type,
                 use_llm_cache,
                 allow_llm_not_valid,
cache_dir
                 ):
    min_max_score = sorted(min_max_score)
    if verbose >= 1:
        print(f"Min max score: {min_max_score}")
    is_valid = False
    max_try = 3
    try_count = 0

    chatgpt_request_dict = (model_name, temperature, top_p_value, max_new_tokens, end_symbol)

    not_valid_count = 0
    while not is_valid:

        if try_count > 0:
            print(f"Not valid! Retry LLM generation. Try count: {try_count}, max try: {max_try}")

        if try_count > max_try:
            print(f"Reach max try")
            sys.exit()

        # END of While Loop
        try_count += 1

        cached_scores = []
        cached_indices = []

        if use_llm_cache:
            non_cached_indices = []
            for i, prompt in enumerate(eval_dim_prompts):
                cache_path = get_llm_response_cache_path(chatgpt_request_dict, prompt, cache_dir)
                if os.path.isfile(cache_path):
                    cached_data = load_save_json(cache_path, 'load', verbose=0)['data']
                    if not isinstance(cached_data, (str, float, int)):
                        non_cached_indices.append(i)
                        os.remove(cache_path)
                    else:
                        cached_indices.append(i)
                        cached_scores.append(cached_data)
                else:
                    non_cached_indices.append(i)
        else:
            non_cached_indices = list(range(len(eval_dim_prompts)))

        llm_eval_dim_scores = np.array([None for _ in range(len(eval_dim_prompts))])
        non_cached_prompts = np.array(eval_dim_prompts)[non_cached_indices]
        non_cached_scores = text_gen_api.llm_gen_concurrent(non_cached_prompts,
                                                            model_name,
                                                            max_try=3,
                                                            verbose=1,
                                                            time_out=40,
                                                            temperature=temperature,
                                                            top_p_value=top_p_value,
                                                            max_new_tokens=max_new_tokens,
                                                            return_time_gap=False,
                                                            max_workers=max_workers,
                                                            stop=end_symbol
                                                            )
        llm_eval_dim_scores[cached_indices] = cached_scores
        llm_eval_dim_scores[non_cached_indices] = non_cached_scores

        if len(non_cached_indices) == 0:
            for x in llm_eval_dim_scores: assert x is not None
        llm_eval_dim_scores = list(llm_eval_dim_scores)

        is_valid = True
        for x_i, x in enumerate(llm_eval_dim_scores):
            if not isinstance(x, str):
                if allow_llm_not_valid and isinstance(x, dict):
                    if 'maximum context length is 4097 tokens' in str(x):
                        llm_eval_dim_scores[x_i] = str(float(np.mean(min_max_score)))
                        not_valid_count += 1
                else:
                    is_valid = False
                    print(f"NOT VALID! {x} is not str!")
                    break
            else:
                if scale_type in {'int', 'float'}:
                    match_num = re.findall(r'^\d+(?:\.\d+)?', x.strip())
                    if len(match_num) == 0:
                        if not allow_llm_not_valid:
                            is_valid = False
                            if verbose >= 1:
                                print("=" * 78)
                                print("RESPONSE NOT VALID!")
                                print("=" * 78)
                                print(f"response: {x}")
                                print(f"prompt: {eval_dim_prompts[x_i]}")
                            break
                        else:
                            llm_eval_dim_scores[x_i] = str(float(np.mean(min_max_score)))
                            not_valid_count += 1
                    else:
                        match_num = [float(x) for x in match_num]
                        match_num = match_num[0]

                        if not (min_max_score[0] <= match_num <= min_max_score[1]):
                            if not allow_llm_not_valid:
                                is_valid = False
                                if verbose >= 1:
                                    print("=" * 78)
                                    print("RESPONSE NOT VALID!")
                                    print("=" * 78)
                                    print(f"response: {x}")
                                    print(f"prompt: {eval_dim_prompts[x_i]}")
                            else:
                                llm_eval_dim_scores[x_i] = str(float(np.mean(min_max_score)))
                                not_valid_count += 1

                elif scale_type in {'choice'}:
                    re_result = re.findall(f'({"|".join(min_max_score)})', x)

                    if len(re_result) != 1:
                        is_valid = False
                        if verbose >= 1:
                            print("=" * 78)
                            print("RESPONSE NOT VALID!")
                            print("=" * 78)
                            print(f"response: {x}")
                            print(f"prompt: {eval_dim_prompts[x_i]}")
                        break
                    else:
                        if re_result[0] not in min_max_score:
                            is_valid = False
                            if verbose >= 1:
                                print("=" * 78)
                                print("RESPONSE NOT VALID!")
                                print("=" * 78)
                                print(f"response: {x}")
                                print(f"prompt: {eval_dim_prompts[x_i]}")
                            break

    if not is_valid:
        raise Exception(f"Exceed max try for getting valid dialogue evaluation dimension scores from llm")
    else:
        if verbose >= 1:
            print("Raw scores: " + str(llm_eval_dim_scores))
        if scale_type in {'int', 'float'}:
            llm_eval_dim_scores = [float(re.findall(r'^\d+(?:\.\d+)?', x.strip())[0]) for x in llm_eval_dim_scores]
        elif scale_type in {'choice'}:
            llm_eval_dim_scores = [re.findall(f'({"|".join(min_max_score)})', x)[0] for x in llm_eval_dim_scores]
        else:
            raise NotImplementedError

        if verbose >= 1:
            print("Cleaned scores: " + str(llm_eval_dim_scores))

    if use_llm_cache:
        for i, prompt in enumerate(non_cached_prompts):
            cache_path = get_llm_response_cache_path(chatgpt_request_dict, prompt, cache_dir)
            if len(non_cached_prompts) == len(set(non_cached_prompts)):
                assert not os.path.isfile(cache_path)
            load_save_json(cache_path, 'save',
                           data={'data': non_cached_scores[i], 'time': str(datetime.datetime.now())}, verbose=0)

    return llm_eval_dim_scores, {'not_valid_count': not_valid_count}

def _get_few_shot_scores(eval_dim, few_shot_xs):
    if eval_dim == 'Uses_Knowledge':
        eval_dim_anno = 'Uses Knowledge'
    elif eval_dim == 'Maintains_Context':
        eval_dim_anno = 'Maintains Context'
    else:
        eval_dim_anno = eval_dim

    few_shot_scores = [float(np.mean(x['annotations'][eval_dim_anno]['score'])) for x in few_shot_xs]
    return few_shot_scores


class FloatClsConverter:
    def __init__(self,
                 all_float_values,
                 bins_method: str = 'auto',
                 is_return_index: bool = False):
        assert isinstance(all_float_values[0], float)
        self.min_value, self.max_value = float(np.min(all_float_values)), float(np.max(all_float_values))
        label_ranges = np.histogram_bin_edges(all_float_values,
                                              bins=bins_method,
                                              range=(min(all_float_values), max(all_float_values)))
        range_size = label_ranges[1] - label_ranges[0]  # calculate range size dynamically
        self.is_return_index = is_return_index
        self._range_dict = {f"{float_v:.2f}-{float_v + range_size:.2f}": (float_v, float_v + range_size) for float_v in
                            label_ranges}
        self._range_index = {f"{float_v:.2f}-{float_v + range_size:.2f}": i for i, float_v in enumerate(label_ranges)}

    @property
    def range_dict(self):
        return self._range_dict

    @property
    def range_index(self):
        return self._range_index

    @property
    def cls_num(self):
        return len(self._range_dict)

    def convert_label(self, float_value, is_strict=False):
        range_label = None
        for tmp_range_label, range_ in self.range_dict.items():
            if range_[0] <= float_value <= range_[1]:
                range_label = tmp_range_label
                break
        if is_strict:
            if range_label is None:
                import ipdb
                ipdb.set_trace()
                raise Exception(f"Strict mode! {float_value} not in range: {self.range_dict}")
            else:
                if self.is_return_index:
                    range_label = self.range_index[range_label]
                return range_label
        else:
            if range_label is None:
                if float_value < self.min_value:
                    return self.convert_label(self.min_value, is_strict=True)
                elif float_value > self.max_value:
                    return self.convert_label(self.max_value, is_strict=True)
                else:
                    raise Exception("Impossible, check bugs in the code.")
            else:
                if self.is_return_index:
                    range_label = self.range_index[range_label]
                return range_label

    def convert_labels(self, float_values):
        return [self.convert_label(x) for x in float_values]
