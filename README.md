# Sensei (先生)
A simple, powerful, minimal codebase to generate synthetic data using OpenAI, MistralAI, AnthropicAI, or offline inference with vLLM

![alt text](Sensei.png)

# Environment set up

Create a virtual environment and install the following packages:

- `pip install --upgrade pip`
- `pip install --no-cache-dir -r ./requirements.txt`

## API text generation

### Choose your provider: OpenAI or MistralAI
- Change `PROVIDER` under `params.py`
- `openai`, `mistral` or `anthropic`

#### For OpenAI
- Change `GPT_MODEL`, `OPENAI_API_KEY` and `OUTPUT_FILE_PATH` under `params.py`

#### For MistralAI
- Change `MISTRALAI_MODEL`, `MISTRALAI_API_KEY` and `OUTPUT_FILE_PATH` under `params.py`

#### For AnthropicAI
- Change `ANTHROPICAI_MODEL`, `ANTHROPICAI_API_KEY` and `OUTPUT_FILE_PATH` under `params.py`

### Run Sensei
- Run with `python main.py`

## Offline text generation with vLLM

### For MistralAI
- Run with `sensei_vllm.py`

This example generates 100 input-output pairs per iteration by using a local instance of `mistralai/Mixtral-8x7B-Instruct-v0.1` for text generation. The script runs an infinite loop and adds samples to an output file after each iteration.

```python
python sensei_vllm.py --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 --backend vllm --tensor-parallel-size 8 --max_len 1024 --dtype float16 --domain lang --outputs ./ --samples_per_iter 100
```

To use the system prompts for code change domain from lang to code (`--domain code`)

# Optional

- Change the topics in `topics.py`
- Change the system contexts in `system_messages.py`
- Change the number of workers in `params.py`