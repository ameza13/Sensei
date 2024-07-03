"""VLLM offline inference pipeline."""

import argparse
import random
from typing import List, Optional, Tuple

from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from vllm import LLM
from vllm import SamplingParams

def prepare_dataset(
    dataset: List,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    print("Start preparing dataset ...")
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    if len(dataset) <= 0: #Nothing to process
        raise ValueError("Nothing to process")
    
    processed_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        prompt = dataset[i] 
        prompt_token_ids = tokenizer(prompt).input_ids # Tokenize the prompts to get prompt_len
        prompt_len = len(prompt_token_ids)
        output_len = fixed_output_len # We use a fixed output_len
        processed_dataset.append((prompt, prompt_len, output_len))
    print("End preparing dataset ...")
    return processed_dataset

class VLLMPipeline:
    def __init__(self,
                model_id: str,
                tokenizer: str,
                quantization: Optional[str],
                tensor_parallel_size: int,
                seed: int,
                n: int,
                use_beam_search: bool,
                trust_remote_code: bool,
                dtype: str,
                max_model_len: Optional[int], # max_new_tokens=None
                enforce_eager: bool,
                kv_cache_dtype: str,
                quantization_param_path: Optional[str],
                device: str,
                enable_prefix_caching: bool,
                enable_chunked_prefill: bool,
                max_num_batched_tokens: int,
                gpu_memory_utilization: float = 0.9,
                download_dir: Optional[str] = None,
                ):
        print("Start model loading ...")
        llm = LLM(
            model=model_id,
            tokenizer=tokenizer,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            quantization_param_path=quantization_param_path,
            device=device,
            enable_prefix_caching=enable_prefix_caching,
            download_dir=download_dir,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        print("End model loading ...")
        self.pipeline = llm
        del llm

        # Params required for inference
        self.use_beam_search = use_beam_search
        self.n = n # Number of responses per prompt
    def __call__(self, 
                 requests: List[Tuple[str, int, int]]):
        prompts = []
        sampling_params = []

        for prompt, _, output_len in requests:
            # Add prompt format
            # prompts.append("".join(["<s>[INST]",prompt, "[/INST]"])) ## Old
            msg_prompt = {'role': 'user', 'content': prompt}
            f_prompt = self.pipeline.llm_engine.tokenizer.tokenizer.apply_chat_template([msg_prompt],
                                                                              tokenize=False, 
                                                                              add_generation_template=True)
            prompts.append(f_prompt)

            # Each prompt may have custom sampling params if needed
            sampling_params.append(
                SamplingParams(
                    n=self.n,
                    temperature=0.8 if self.use_beam_search else 1.0, 
                    top_p=0.9,
                    use_beam_search=self.use_beam_search,
                    ignore_eos=False, 
                    max_tokens=output_len,
                ))

        assert len(prompts) == len(sampling_params)
        outputs = self.pipeline.generate(prompts, sampling_params, use_tqdm=True)
        prompts = []
        responses = []
        for output in outputs:
            prompt = output.prompt
            response = output.outputs[0].text
            response = response.replace(prompt, '').strip() # This line cleans prompt leak
            prompts.append(prompt)
            responses.append(response)
            # print(f"Prompt: {prompt!r}, Generated text: {response!r}") # TEST
        return prompts,responses

# This is where inference is called from, migrate this part to WizardLM class
def pipeline_setup(args: argparse.Namespace):
    print(args) # TEST
    random.seed(args.seed) # delete?

    llm_pipeline = None

    if args.backend == "vllm":
        llm_pipeline = VLLMPipeline(
                args.model_id, args.tokenizer, args.quantization,
                args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
                args.trust_remote_code, args.dtype, args.max_model_len,
                args.enforce_eager, args.kv_cache_dtype,
                args.quantization_param_path, args.device,
                args.enable_prefix_caching, args.enable_chunked_prefill,
                args.max_num_batched_tokens, args.gpu_memory_utilization,
        )
    elif args.backend == "hf":
        # assert args.tensor_parallel_size == 1
        # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
        # TO DO: Implement HFPipeline class
        llm_pipeline = None
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    return llm_pipeline
    