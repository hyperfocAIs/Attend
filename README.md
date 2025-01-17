Our attention is being stolen.<sup>[1](https://www.theguardian.com/science/2022/jan/02/attention-span-focus-screens-apps-smartphones-social-media)</sup>

**TAKE. IT. BACK.**

Attend - to what matters.

# Overview

Attend is a project born of the idea that our technology should help people spend their time, engergy and attention on what they value most, and that, far too often, it does the opposite.

Start Attend. Tell it what you want to do. It will help you do it, and not do something else.

# How it works

Attend is a voice asssitant powered by generative AI (Voice activity detection, Speech to text, Text to speech, and Large language models) that asks you what you want to do, and then helps you do it. This includes processing screenshots from your computer and determining if what you're doing is aligned with what you said you wanted to do.

# How to use it

Attend is under early and active development, there are likely bugs, and features will be added and improved frequently.

## AI models

Attend requires API access to: TTS, STT, a text only LLM, and a vision LLM. You can use the same model for the last two if you want, but may want to use a different model for the text only LLM if the vision LLM isn't good for dialogue or driving Attend's agentic features.

Here are examples I used to set everything up locally:


### Vision LLM
```
docker run --gpus '"device=0"' --runtime nvidia -v ~/.cache/huggingface:/root/.cache/huggingface -p 8002:8002 --ipc=host --name serve_vision --restart always vllm/vllm-openai:v0.6.6.post1 --model Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8 -tp 1 --host 0.0.0.0 --port 8002 --gpu-memory-utilization 0.7 --max-model-len 24000 --max-num-seqs 4 --enable-prefix-caching --guided-decoding-backend outlines
```

### Text LLM
```
docker run --gpus '"device=1,2,3,4"' --runtime nvidia -v ~/.cache/huggingface:/root/.cache/huggingface -p 2243:2243 --ipc=host --name serve_text --restart always vllm/vllm-openai:v0.6.6.post1 --model CalamitousFelicitousness/Llama-3.3-70B-Instruct-W8A8-INT8 -tp 4 --host 0.0.0.0 --port 2243 --gpu-memory-utilization 0.97 --max-model-len 32688 --max-num-seqs 4 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser llama3_json --tool-call-parser llama3_json --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

### TTS

```
git clone https://github.com/remsky/Kokoro-FastAPI.git

cd Kokoro-FastAPI

docker compose up --build
```

### STT

```
huggingface-cli download deepdml/faster-whisper-large-v3-turbo-ct2

docker run --gpus '"device=0"' --runtime nvidia --publish 8001:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface --env WHISPER__MODEL=deepdml/faster-whisper-large-v3-turbo-ct2 --env WHISPER__TTL=-1 --name attend_stt --restart always fedirz/faster-whisper-server:sha-caba05a-cuda
```

## Your computer

Copy the repo, setup a virtual environment, `pip install -r requirements.txt`, rename attend_config-example.yaml to attend_config.yaml and edit it to use the APIs/model you want, and then run `python main.py`.

