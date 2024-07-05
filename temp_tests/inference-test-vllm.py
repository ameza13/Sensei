import torch
from vllm import LLM, SamplingParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=8)
# # llm = LLM(model="mistralai/Mixtral-8x22B-Instruct-v0.1", tensor_parallel_size=8)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=False, max_tokens = 1024)

# Simple prompts
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]

# Chat prompts
# Issue: Mistral models do not support system prompt: https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF
# REFERENCE: https://github.com/vllm-project/vllm/issues/3119
# REFERENCE: https://github.com/huggingface/transformers/issues/27922
# MIXTRAL DOES NOT ACCEPTE SYSTEM PROMPT, WORK AROUND: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/69
"""
same way as Llama2 ? -> no, it seems for mixtral system prompt must be part of [INST] system prompt [/INST]</s>

[INST] <>
{your_system_message}
<>

{user_message_1} [/INST]
"""

# For instruction generation simply pre-pending the system prompt to user prompt should be fine:
PROMPT_1 = """For the following SUBJECT_AREA, generate a question that covers a very narrow topic in the SUBJECT_AREA, with sufficient depth and breadth. The topic in the question should be important to the SUBJECT_AREA, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question."""
msg_system = {'role': 'system', 'content': str(PROMPT_1)}
msg_user = {'role': 'user', 'content': 'SUBJECT_AREA: Science - physics, chemistry, biology, astronomy, etc.'} 
msg = f"{msg_system['content']}\n{msg_user['content']}"
msg_prompt = {'role': 'user', 'content': msg.strip()}

messages = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template([msg_prompt], tokenize=False, add_generation_template=True)]
outputs = llm.generate(messages, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated Instruction (chat template): {generated_text!r}")

# messages = "".join(["<s>[INST]",msg_system['content'],"\n",msg_prompt['content'], "[/INST]"])
# outputs = llm.generate(messages, sampling_params)

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}\nGenerated Instruction (without chat template): {generated_text!r}")

instruction = generated_text
# For response generation order depends on the system_message:
# SYSTEM_MESSAGES_ORCA -> convenient to pre-pend
msg_system = {'role': 'system', 'content': "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer."}
msg_user = {'role': 'user', 'content': instruction} 
msg = f"{msg_system['content']}\n{msg_user['content']}"
msg_prompt_orca = {'role': 'user', 'content': msg.strip()}

messages = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template([msg_prompt_orca], tokenize=False, add_generation_template=True)]
outputs = llm.generate(messages, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated Response (sys msg before): {generated_text!r}")

# SYSTEM_MESSAGES_TESS -> convenient to post-pend
msg_system = {'role': 'system', 'content': """
    Answer the Question in a logical, step-by-step manner that makes the reasoning process clear.
    First, carefully analyze the question to identify the core issue or problem to be solved. This helps frame the foundation for logical reasoning.
    Next, break down the issue into sub-components and constraints that need to be addressed. This allows tackling the problem in a structured way.
    For each sub-component, leverage the knowledge and inference skills to generate multiple hypotheses or possibilities that could lead to a solution.
    Critically evaluate each hypothesis based on validity, relevance to the question, and how well it addresses the sub-component when logically combined with other steps.
    Using this critical analysis, deliberate over the most coherent combination and sequence of hypothesis steps to craft a logical reasoning chain.
    Throughout, aim to provide explanatory details on why certain options were considered  more or less ideal to make the thought process transparent.
    If it was determined that there is a gap in the reasoning chain, backtrack and explore alternative hypotheses to plug the gap until there is a complete logical flow.
    Finally, synthesize the key insights from the reasoning chain into a concise answer that directly addresses the original question.
    In summary, leverage a structured, critical thinking process with iterative refinement to demonstrate strong logical reasoning skills in the answer.
    """}

msg_user = {'role': 'user', 'content': instruction} 
msg = f"{msg_user['content']}\n{msg_system['content']}"
msg_prompt_tess = {'role': 'user', 'content': msg.strip()}

messages = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template([msg_prompt_tess], tokenize=False, add_generation_template=True)]
outputs = llm.generate(messages, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated Response (sys msg after): {generated_text!r}")