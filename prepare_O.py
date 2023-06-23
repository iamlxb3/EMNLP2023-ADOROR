import copy
import torch
from tqdm import tqdm
from utils import FloatClsConverter
from utils import _get_few_shot_scores
from utils import score_by_llm
from dialogue_eval_dimension_prompt import DialogueEvalDimensionPrompt


def _embedding_demo_selector_inputs(sentence_embedder,
                                    score_cls_converter,
                                    all_raw_prompts,
                                    infer_text,
                                    all_demos,
                                    all_scores,
                                    cv_i,
                                    eval_dim,
                                    is_train=False
                                    ):
    # get all demos embedding
    demo_few_shot_k = len(all_demos[0])
    all_demos_flatten_raw = [y for x in all_demos for y in x]
    all_demos_flatten = sentence_embedder.embed(all_demos_flatten_raw,
                                                f'{eval_dim} Cv-i {cv_i}, Get all demo embedding')
    all_demos = all_demos_flatten.reshape(int(all_demos_flatten.shape[0] / demo_few_shot_k),
                                          demo_few_shot_k,
                                          all_demos_flatten.shape[-1])

    # get all prompts embedding
    all_prompts = sentence_embedder.embed(all_raw_prompts, f'{eval_dim} Cv-i {cv_i}, Get raw prompt embedding')

    # get infer text embedding
    if is_train:
        infer_text_embed = sentence_embedder.embed(infer_text, f'{eval_dim} Cv-i {cv_i}, Get infer text embedding')
    else:
        infer_text_embed = sentence_embedder.embed([infer_text], verbose=False)

    all_score_indices = []
    for score_tuple in all_scores:
        score_indices = tuple(score_cls_converter.convert_labels(score_tuple))
        all_score_indices.append(score_indices)
    all_score_indices = torch.Tensor(all_score_indices).long()
    if is_train:
        assert len(all_score_indices) == len(all_demos) == len(all_prompts) == len(infer_text_embed)
    else:
        assert len(all_score_indices) == len(all_demos) == len(all_prompts)
    return all_demos, all_score_indices, all_prompts, infer_text_embed


def create_data_for_demo_selector_one_prompt(auged_prompt,
                                             repeat_n,
                                             few_shot_k,
                                             eval_dim_config,
                                             is_normalize_score,
                                             data,
                                             to_sample_data,
                                             few_shot_retriever,
                                             eval_dim,
                                             text_gen_api,
                                             model_name,
                                             temperature,
                                             top_p_value,
                                             max_new_tokens,
                                             max_workers,
                                             end_symbol,
                                             use_llm_cache,
                                             allow_llm_not_valid,
                                             sample_textualizer):

    eval_dim_config = copy.deepcopy(eval_dim_config)
    eval_dim_config[eval_dim]['prompt'] = auged_prompt
    eval_dim_prompter = DialogueEvalDimensionPrompt(eval_dim,
                                                    few_shot_k,
                                                    eval_dim_config[eval_dim],
                                                    eval_dim_config['General'],
                                                    is_normalize_score=is_normalize_score,
                                                    normalize_score_max=10)

    llm_inputs = []
    infer_texts = []
    human_scores = []
    demos = []
    demo_scores = []

    for _ in range(repeat_n):
        for test_x_i, test_x in enumerate(data):

            if test_x in to_sample_data:
                mask = to_sample_data != test_x
                candidate_train_data = to_sample_data[mask]
            else:
                candidate_train_data = to_sample_data

            few_shot_xs = few_shot_retriever.retrieve_few_shot_samples(candidate_train_data, test_x, 'context')

            infer_text = sample_textualizer.sample_textualize(test_x)
            few_shot_contexts = sample_textualizer.samples_textualize(few_shot_xs)
            few_shot_scores = _get_few_shot_scores(eval_dim, few_shot_xs)

            demos.append(tuple(few_shot_contexts))
            demo_scores.append(tuple(few_shot_scores))
            infer_texts.append(infer_text)

            # fill input into prompt
            few_shot_demo_str = eval_dim_prompter.fill_few_shot_demonstrations(few_shot_xs)

            # fill input x
            infer_sample_str = eval_dim_prompter.fill_one_sample(test_x).strip()

            # filled prompt
            eval_dim_prompt_x = eval_dim_prompter.fill_dimension_prompt(few_shot_demo_str,
                                                                        infer_sample_str)

            llm_inputs.append(eval_dim_prompt_x)

            # get the humman score
            score = eval_dim_prompter.get_score_from_anno_dict(test_x['annotations'])
            human_scores.append(score)

    scale_type = eval_dim_prompter.scale_type
    if eval_dim_prompter.is_choice_to_score is True:
        scale_type = 'float'

    llm_eval_dim_scores, llm_result_meta = score_by_llm(llm_inputs,
                                                        text_gen_api,
                                                        model_name,
                                                        temperature,
                                                        top_p_value,
                                                        max_new_tokens,
                                                        max_workers,
                                                        end_symbol,
                                                        0,
                                                        eval_dim_prompter.min_max_score,
                                                        scale_type,
                                                        use_llm_cache,
                                                        allow_llm_not_valid,
                                                        None
                                                        )
    for x in llm_eval_dim_scores: assert isinstance(x, float)
    for x in human_scores: assert isinstance(x, float)

    if scale_type in {'choice'} and eval_dim_prompter.is_choice_to_score is False:
        llm_eval_dim_scores = [eval_dim_prompter.scale_mapping[x] for x in llm_eval_dim_scores]
        human_scores = [eval_dim_prompter.scale_mapping[x] for x in human_scores]
        print(f"LLM scores: {llm_eval_dim_scores}")
    assert len(llm_eval_dim_scores) == len(human_scores)

    return llm_eval_dim_scores, human_scores, demos, demo_scores, infer_texts


def prepare_train_data_for_demo_selector(demo_select_prompts,
                                         demo_select_repeat_n,
                                         few_shot_k,
                                         origin_eval_dim_config,
                                         is_normalize_score,
                                         train_subset,
                                         data,
                                         few_shot_retriever,
                                         eval_dim,
                                         text_gen_api,
                                         model_name,
                                         temperature,
                                         top_p_value,
                                         max_new_tokens,
                                         max_workers,
                                         end_symbol,
                                         use_llm_cache,
                                         allow_llm_not_valid,
                                         sample_textualizer,
                                         sentence_embedder,
                                         cv_i
                                         ):
    demo_select_prompts = copy.deepcopy(demo_select_prompts)
    demo_select_tqdm = tqdm(demo_select_prompts, total=len(demo_select_prompts), colour='YELLOW',
                            desc=f'CV-{cv_i} Preparing demo selector training data')

    all_scores = []
    all_demos = []
    all_infer_texts_raw = []
    all_label_Y = []
    all_raw_prompts = []

    for _, _, _, auged_prompt in demo_select_tqdm:
        llm_eval_dim_scores, human_scores, demos, demo_scores, infer_texts = create_data_for_demo_selector_one_prompt(
            auged_prompt,
            demo_select_repeat_n,
            few_shot_k,
            origin_eval_dim_config,
            is_normalize_score,
            train_subset,
            data,
            few_shot_retriever,
            eval_dim,
            text_gen_api,
            model_name,
            temperature,
            top_p_value,
            max_new_tokens,
            max_workers,
            end_symbol,
            use_llm_cache,
            allow_llm_not_valid,
            sample_textualizer)
        for x in llm_eval_dim_scores: assert isinstance(x, float)
        for x in human_scores: assert isinstance(x, float)
        label_Y = torch.abs(torch.Tensor(llm_eval_dim_scores) - torch.Tensor(human_scores))
        all_scores.extend(demo_scores)
        all_demos.extend(demos)
        all_infer_texts_raw.extend(infer_texts)
        all_label_Y.extend(label_Y.tolist())
        all_raw_prompts.extend([auged_prompt.split('\n\n')[1].split('\n')[0] for _ in range(len(label_Y))])

    all_label_Y = torch.Tensor(all_label_Y)

    all_label_Y = 1 / (torch.e ** all_label_Y)

    # init float to cls converter
    score_cls_converter = FloatClsConverter([x for x_tuple in all_scores for x in x_tuple],
                                            bins_method='auto',
                                            is_return_index=True)

    all_demos, all_score_indices, all_prompts, all_infer_texts = _embedding_demo_selector_inputs(sentence_embedder,
                                                                                                 score_cls_converter,
                                                                                                 all_raw_prompts,
                                                                                                 all_infer_texts_raw,
                                                                                                 all_demos,
                                                                                                 all_scores,
                                                                                                 cv_i,
                                                                                                 eval_dim,
                                                                                                 is_train=True)
    assert len(all_label_Y) == len(all_demos)

    return all_demos, all_score_indices, all_prompts, all_infer_texts_raw, all_infer_texts, all_label_Y, \
           score_cls_converter
