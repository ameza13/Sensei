import os
import json
import numpy as np
import uuid
import time
import argparse
from transformers import (AutoTokenizer)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from system_messages import (
    SYSTEM_MESSAGES_ORCA,
    SYSTEM_MESSAGES_TESS,
    SYSTEM_MESSAGES_CODE,
)
from topics import ALL_TOPICS, TOPICS_1_5, TOPICS_6_10, TOPICS_11_15, TOPICS_16_20, TOPICS_21_25, TOPICS_26_30, TOPICS_31_35, TOPICS_35_40, TOPICS_41_45, TOPICS_45_50

from pipelines import pipeline_setup, prepare_dataset, VLLMPipeline

"""
python sensei_vllm.py --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 --backend vllm --tensor-parallel-size 8 --min_len 1 --max_len 7698 --dtype float16 --domain lang --outputs /path/to/save/dataset --samples_per_iter 2
"""

# CREATE INSTRUCTION PROMPTS
def create_instruction_prompt(
    topic_selected,
    system_message_generation
):
    msg_system = str(system_message_generation)
    msg_user = f'SUBJECT_AREA: {topic_selected}'
    msg_prompt = "".join([msg_system," ",msg_user])
    return msg_prompt.strip()

# CREATE RESPONSE PROMPTS
def create_response_prompt(
    system_message_selected,
    instruction:str,
    pre_pend_sys_msg:bool=True
):
    msg_system = str(system_message_selected)
    msg_user = instruction

    if pre_pend_sys_msg:
        msg_prompt = "".join([msg_system," ",msg_user])
    else:
        msg_prompt = "".join([msg_user, " ", msg_system])
    return msg_prompt.strip()

def sensei(llm_pipeline, 
           tokenizer,
           trust_remote_code:bool,
           num_instances:int, 
           output_file_path:str, 
           system_message_generation:str, 
           system_messages:list,
           topics_group:str):
    prompts_instructions = []
    prompts_responses = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)

    # Create instruction prompts
    if topics_group == "A":
        TOPICS = TOPICS_1_5
    elif topics_group == "B":
        TOPICS = TOPICS_6_10
    elif topics_group == "C":
        TOPICS = TOPICS_11_15
    elif topics_group == "D":
        TOPICS = TOPICS_16_20
    elif topics_group == "E":
        TOPICS = TOPICS_21_25
    elif topics_group == "F":
        TOPICS = TOPICS_26_30
    elif topics_group == "G":
        TOPICS = TOPICS_31_35
    elif topics_group == "H":
        TOPICS = TOPICS_35_40
    elif topics_group == "E":
        TOPICS = TOPICS_41_45
    elif topics_group == "J":
        TOPICS = TOPICS_45_50
    else:
        TOPICS = ALL_TOPICS
    
    print("= TOPICS =")
    for topic in TOPICS:
        print(topic)

    for _ in range(num_instances):
        topic_number = np.random.randint(0, len(TOPICS))
        topic_selected = TOPICS[topic_number]
        prompts_instructions.append(create_instruction_prompt(topic_selected=topic_selected,
                                                              system_message_generation=system_message_generation)) #PROMPT_1
    # Create instructions
    requests = prepare_dataset(dataset=prompts_instructions, 
                               tokenizer=tokenizer, 
                               fixed_output_len=max_len)
    assert len(requests) == len(prompts_instructions)

    print(f'Creating {len(requests)} instructions ...')
    t0 = time.time()
    prompts_inst, instructions = llm_pipeline(requests) # Inference only returns the response
    t1 = time.time()
    print("Creating instructions took %.4f seconds" % (t1 - t0))    

    assert len(requests) == len(instructions)

    # TEST
    # for prompt, instruction in zip(prompts_inst,instructions):
    #     print(f"Prompt: {prompt!r}\nInstruction: {instruction!r}")

    # Create response prompts
    for instruction in instructions:
        if code_data:
            pre_pend_sys_msg = False
            system_message_number = np.random.randint(0, len(system_messages)) # SYSTEM_MESSAGES
            system_message_selected = system_messages[system_message_number]
        else:
            system_message_number = np.random.randint(0, len(system_messages))
            pre_pend_sys_msg = (system_message_number > 7) # pre-pend system message
            # SYSTEM_MESSAGES_TESS 0-7 -> 8 messages
            # SYSTEM_MESSAGES_ORCA 0-15 -> 16 messages
            system_message_selected = system_messages[system_message_number]
        
        # TEST
        # print(f"system_message_number: {system_message_number}")
        # print(f"pre_pend_sys_msg: {pre_pend_sys_msg}")

        prompts_responses.append(
            create_response_prompt(system_message_selected=system_message_selected,
                                    instruction=instruction,
                                    pre_pend_sys_msg=pre_pend_sys_msg))
    # Create prompts
    requests = prepare_dataset(dataset=prompts_responses, 
                               tokenizer=tokenizer, 
                               fixed_output_len=max_len)
    assert len(requests) == len(prompts_responses)

    print(f'Creating {len(requests)} responses ...')
    t0 = time.time()
    prompts_res, responses = llm_pipeline(requests) # Inference only returns the response
    t1 = time.time()
    print("Creating responses took %.4f seconds" % (t1 - t0))    

    assert len(requests) == len(responses)

    # TEST
    # for prompt, response in zip(prompts_res,responses):
    #     print(f"Prompt: {prompt!r}\nResponse: {response!r}")   

    assert len(prompts_inst) == len(prompts_res) == len(instructions) == len(responses)

    # Update output file
    valid_samples = 0
    for prompt_inst,instruction, prompt_res, response in zip(prompts_inst,instructions,prompts_res,responses):
        if len(instruction)>0 and len(response)>0: # We only save the sample if there is no empty instruction or response
            instance = {"input_prompt":prompt_inst,"input":instruction,"output_prompt":prompt_res,"output": response}
            with open(output_file_path, "a") as output_file:
                output_file.write(json.dumps(instance) + "\n")
            valid_samples +=1
        else:
            print(f"Skipping Sample\n\tInstruction:{instruction}\n\tResponse:{response}")
    print(f"{str(valid_samples)} valid samples out of {len(instructions)} total samples were added to: {output_file_path}")
    return valid_samples

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sensei")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf"],
                        default="vllm")
    parser.add_argument("--dataset", # TO DO: It must also work for vocab files
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len", # Note: Need to set when data comes from vocab
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len", # Note: Need to set when data comes from vocab
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model-id", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="HF model id (i.e., mistralai/Mixtral-8x7B-Instruct-v0.1).") # --model
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--hf-max-batch-size", # Note: Needed for HF backend
    #                     type=int,
    #                     default=None,
    #                     help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code', # Not sure what to send here
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], #float16 or bfloat16
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager", # Not sure what to send here
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    # SENSEI Params
    parser.add_argument("--samples_per_iter",
                    type=int,
                    required=False,
                    default=10,
                    help="Number of samples to be generated.")
    parser.add_argument("--min_len",
                    type=int,
                    required=False,
                    default=512,
                    help="Lower limit for prompt length.")
    parser.add_argument("--max_len",
                    type=int,
                    required=False,
                    default=1024,
                    help="Upper limit for prompt length.")
    # parser.add_argument("--verbose",
    #                 type=bool,
    #                 required=False,
    #                 default=False,
    #                 help="Whether to enable verbose printing.")
    parser.add_argument("--outputs",
                        type=str,
                        required=False,
                        default="./", 
                        help="Directory to store output files")
    parser.add_argument("--domain",
                        type=str,
                        required=False,
                        default="lang", 
                        help="The domain of the data. Options: lang, code")
    parser.add_argument("--topics_group",
                        type=str,
                        required=False,
                        default="ALL", 
                        help="e.g., (A, B, C, D, ALL, etc.)")
    return parser.parse_args()

if __name__ == "__main__":
    # Inference pipeline setup
    args = parse_arguments()
    if args.tokenizer is None:
        args.tokenizer = args.model_id   

    samples_per_iter = args.samples_per_iter # will finish a batch of input-output pairs
    output_file_path = args.outputs
    code_data = (args.domain == "code")
    max_len = args.max_len
    tokenizer = args.tokenizer
    trust_remote_code = args.trust_remote_code
    topics_group = args.topics_group

    if code_data:
        SYSTEM_MESSAGES = SYSTEM_MESSAGES_CODE
        PROMPT_1 = """For the following SUBJECT_AREA, generate a question that covers a very narrow topic in the SUBJECT_AREA, with sufficient depth and breadth. The topic in the question should be important to the SUBJECT_AREA, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question. The question should seek an answer with computer code."""

    else:
        SYSTEM_MESSAGES = SYSTEM_MESSAGES_TESS + SYSTEM_MESSAGES_ORCA
        PROMPT_1 = """For the following SUBJECT_AREA, generate a question that covers a very narrow topic in the SUBJECT_AREA, with sufficient depth and breadth. The topic in the question should be important to the SUBJECT_AREA, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question."""

    print("Downloading model to run locally ...")
    llm_pipeline = pipeline_setup(args)

    # Output file
    group_id = str(uuid.uuid4())[:4]
    output_file_path = os.path.join(output_file_path, f"sensei-{group_id}.jsonl")
    # print(output_file_path) # TEST

    iteration = 0
    total_samples=0
    while True:
        # start_time = time.time()
        
        print(f"Generating synthetic data {str(iteration)}...")
        new_samples = sensei(llm_pipeline=llm_pipeline,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            num_instances=samples_per_iter,
            output_file_path=output_file_path,
            system_message_generation=PROMPT_1,
            system_messages=SYSTEM_MESSAGES,
            topics_group=topics_group)
        total_samples += new_samples
        print(f"\tTotal samples estimate: {str(total_samples)}...")
        iteration +=1

        # final_time = time.time() - start_time
        # print(f'All Computation complete, total run took {final_time:.2f}s')