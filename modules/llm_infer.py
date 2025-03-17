# -*- coding: utf-8 -*-
# @Time       : 2025/2/21 11:11
# @Author     : Marverlises
# @File       : llm_infer.py
# @Description: Module for large language model inference
import json
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from openai import OpenAI
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# Configure GPU if available
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class LLMInfer:
    """
    Class for various LLM inference methods.
    Supports local model inference via vLLM and API inference via OpenAI-compatible endpoints.
    """

    @staticmethod
    def vllm_infer(
            prompts: List[str],
            model_path: str,
            generation_config: Optional[str] = None,
            sampling_params: Optional[Any] = None,
            max_tokens: int = 16384,
            temperature: float = 0.6,
            top_p: float = 0.95,
            stop_token_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text using local models through vLLM.
        
        Args:
            prompts (List[str]): List of prompts to process
            model_path (str): Path to the local model
            generation_config (str, optional): Path to generation config. Defaults to None.
            sampling_params (Any, optional): Sampling parameters. Defaults to None.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 16384.
            temperature (float, optional): Sampling temperature. Defaults to 0.6.
            top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
            stop_token_ids (List[int], optional): List of token IDs that stop generation. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: List of responses with 'prompt' and 'generated_text' fields
        """
        if LLM is None:
            raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")

        try:
            logger.info(f"Initializing vLLM with model: {model_path}")
            llm = LLM(model=model_path, generation_config=generation_config)

            # Create sampling parameters if not provided
            if sampling_params is None:
                stop_tokens = stop_token_ids or [151643]  # Default stop token for some models
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    stop_token_ids=stop_tokens,
                    max_tokens=max_tokens
                )

            logger.info(f"Generating with vLLM for {len(prompts)} prompts")
            outputs = llm.generate(prompts, sampling_params)

            # Process outputs
            response_info = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                response_info.append({
                    'prompt': prompt,
                    'generated_text': generated_text
                })

            return response_info
        except Exception as e:
            logger.error(f"Error in vLLM inference: {e}")
            # Return error responses rather than failing completely
            return [{'prompt': p, 'generated_text': f"Error in generation: {str(e)}", 'error': True} for p in prompts]

    @staticmethod
    def API_infer(
            api_secret_key: str,
            base_url: str,
            queries: List[str],
            model: str,
            sys_message: str = "",
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate text through API inference with OpenAI-compatible endpoints.
        
        Args:
            api_secret_key (str): API key for authentication
            base_url (str): Base URL for the API endpoint
            queries (List[str]): List of queries to process
            model (str): Model identifier to use
            sys_message (str, optional): System message for chat models. Defaults to "".
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to None.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            List[Dict[str, str]]: List of responses with 'prompt' and 'generated_text' fields
        """
        if OpenAI is None:
            raise ImportError("OpenAI client is not installed. Please install it with 'pip install openai'.")

        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=api_secret_key, base_url=base_url)

            # Default system message if empty
            system_message = sys_message or "You are a helpful assistant."

            response_info = []

            for query in queries:
                try:
                    # Create request parameters
                    params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": query}
                        ],
                        "temperature": temperature,
                        "stream": stream
                    }

                    # Add max_tokens if specified
                    if max_tokens:
                        params["max_tokens"] = max_tokens

                    # Make the API call
                    resp = client.chat.completions.create(**params)

                    if stream:
                        # Handle streaming responses
                        full_content = ""
                        for chunk in resp:
                            if chunk.choices and len(chunk.choices) > 0:
                                content = chunk.choices[0].delta.content
                                if content:
                                    full_content += content
                        response_info.append({'prompt': query, 'generated_text': full_content})
                    else:
                        # Handle regular responses
                        response_info.append({
                            'prompt': query,
                            'generated_text': resp.choices[0].message.content
                        })

                except Exception as e:
                    logger.error(f"Error in API inference for query: {e}")
                    response_info.append({
                        'prompt': query,
                        'generated_text': f"Error in generation: {str(e)}",
                        'error': True
                    })

            return response_info
        except Exception as e:
            logger.error(f"Error initializing API client: {e}")
            return [{'prompt': q, 'generated_text': f"Error in API connection: {str(e)}", 'error': True} for q in
                    queries]

    @classmethod
    def infer(cls,
              prompts: List[str],
              model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generic inference method that selects the appropriate inference method based on the model configuration.
        
        Args:
            prompts (List[str]): List of prompts to process
            model_config (Dict[str, Any]): Model configuration with details about the model and inference method
            
        Returns:
            List[Dict[str, Any]]: List of responses with 'prompt' and 'generated_text' fields
        """
        # Extract inference method
        infer_method = model_config.get('method', 'api')

        if infer_method.lower() == 'vllm':
            # Extract vLLM params
            model_path = model_config.get('model_path')
            if not model_path:
                raise ValueError("model_path must be provided for vLLM inference")

            generation_config = model_config.get('generation_config')
            sampling_params = model_config.get('sampling_params')
            max_tokens = model_config.get('max_tokens', 16384)
            temperature = model_config.get('temperature', 0.6)
            top_p = model_config.get('top_p', 0.95)
            stop_token_ids = model_config.get('stop_token_ids')

            return cls.vllm_infer(
                prompts=prompts,
                model_path=model_path,
                generation_config=generation_config,
                sampling_params=sampling_params,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids
            )
        else:  # Default to API
            # Extract API params
            api_key = model_config.get('api_key')
            if not api_key:
                raise ValueError("api_key must be provided for API inference")

            base_url = model_config.get('base_url', 'https://api.openai.com/v1')
            model = model_config.get('model', 'gpt-3.5-turbo')
            system_message = model_config.get('system_message', '')
            temperature = model_config.get('temperature', 0.7)
            max_tokens = model_config.get('max_tokens')
            stream = model_config.get('stream', False)

            return cls.API_infer(
                api_secret_key=api_key,
                base_url=base_url,
                queries=prompts,
                model=model,
                sys_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )


if __name__ == '__main__':
    # read the json file
    file_path = "/ai/teacher/mwt/code/by/project/Paper_Agent/pdf_analyzer/test.json"
    # read the data as json
    with open(file_path) as json_file:
        data = json.load(json_file)

    text_data = [item['type'] + ":" + item['text'] for item in data]
    prompt_list = ["请讲一下这篇论文" + str(text_data)]
    response = LLMInfer.infer(prompt_list, model_config={
        'method': 'vllm',
        'model_path': '/ai/teacher/mwt/code/by/models/deepseek-14b',
        'generation_config': '/ai/teacher/mwt/code/by/models/deepseek-14b/',
        'sampling_params': None
    })
    for resp in response:
        print("Prompt:", resp['prompt'])
        print("Generated Text:", resp['generated_text'])
