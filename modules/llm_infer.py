# -*- coding: utf-8 -*-
# @Time       : 2025/2/21 11:11
# @Author     : Marverlises
# @File       : llm_infer.py
# @Description: PyCharm

import os

from openai import OpenAI
from vllm import LLM, SamplingParams
from typing import List


class LLMInfer:
    @staticmethod
    def vllm_infer(prompts: List[str], model_path: str, generation_config: str,
                   sampling_params: SamplingParams | None) -> \
            list[dict[str, str | None]]:
        """
        This function is used to generate text from the prompt list.
        :param prompts:             The list of prompts.
        :param model_path:          The path of the model.
        :param generation_config:   The path of the generation config.
        :param sampling_params:     The parameters of sampling.
        :return:                    The list of generated text.
        """

        llm = LLM(model=model_path, generation_config=generation_config)

        # The Parameters of SamplingParams is from the generation_config of the model.
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, stop_token_ids=[151643], max_tokens=16384)
        outputs = llm.generate(prompts, sampling_params)
        response_info = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            response_info.append({'prompt': prompt, 'generated_text': generated_text})

        return response_info

    @staticmethod
    def API_infer(api_secret_key: str, base_url: str, queries: str, model: str, sys_message: str) -> list[
        dict[str, str]]:
        """
        This function is used to generate text from the API request.
        :param queries:             The list of queries.
        :param api_secret_key:      The API secret key.
        :param base_url:            The base URL.
        :param model:               The model name.
        :param sys_message:         The system message.
        :return:                    The list of generated text.
        """
        client = OpenAI(api_key=api_secret_key, base_url=base_url)
        response_info = []

        for query in queries:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": query}
                ]
            )
            response_info.append({'prompt': query, 'generated_text': resp.choices[0].message.content})

        return response_info


if __name__ == '__main__':
    prompt_list = [
        "你是谁？",
        "美国的首都是哪？",
        "你是如何认识未来的AI发展的？",
    ]
    response = LLMInfer.vllm_infer(prompt_list, model_path="/ai/teacher/mwt/code/by/models/deepseek-14b",
                                   generation_config="/ai/teacher/mwt/code/by/models/deepseek-14b/",
                                   sampling_params=None)
    for resp in response:
        print("Prompt:", resp['prompt'])
        print("Generated Text:", resp['generated_text'])
