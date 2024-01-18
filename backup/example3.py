import torch
import intel_extension_for_pytorch as ipex

from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer

import time
import numpy as np
#from benchmark_util import BenchmarkWrapper
from transformers import TextIteratorStreamer
def stream_chat(model, tokenizer, prompt, max_new_tokens, history=[], device="xpu"):
    # format conversation context as prompt through chat history
    #prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=input_str)
    #prompt = LLAMA2_PROMPT_FORMAT.format(prompt=input)
    input_ids = tokenizer([prompt], return_tensors='pt').to(device)
    print("stream_chat-----input_ids:", input_ids)

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True, # skip prompt in the generated tokens
                                    skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens
    )

    # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
    from threading import Thread

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    history = []

    output_str = ""
    for stream_output in streamer:
        output_str += stream_output
        yield output_str, history


model_path = "./models/chatglm3-6b-int4"   ## pass
#model_path = r"C:\\Users\\MTL822\\Documents\\audiollm\\models\\chatglm3-6b-int4"   ## pass
#model_path = r"C:\\Users\\MTL822\\Documents\\audiollm\\models\\Qwen-7B-Chat-int4"
#model_path = r"C:\\Users\\MTL822\\Documents\\audiollm\\models\\AquilaChat2-7B-int4"
#model_path = "./models/Baichuan2-7B-Chat-int4"
#model_path = r"C:\\Users\\Administrator\\Documents\\LLM_Demo\\checkpoint\\CodeShell-7B-int4"
#model_path = r"C:\\Users\\Administrator\\Documents\\LLM_Demo\\checkpoint\\rwkv-4-world-7b"
#model_path = r"C:\\Users\\MTL822\\Documents\\audiollm\\models\\internlm-chat-7b-8k-int4"

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, optimize_model=True, load_in_4bit=True).eval()
#model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True, optimize_model=True).eval()

#model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,low_cpu_mem_usage=True, torch_dtype=torch.half).eval()
#model.rwkv._rescale_layers()

#model = AutoModel.from_pretrained(model_path, trust_remote_code=True, optimize_model=False, load_in_4bit=True).eval()
model = AutoModel.load_low_bit(model_path, trust_remote_code=True, optimize_model=True).eval()

if 0:
    model.save_low_bit(model_path +  "-int4/")
    tokenizer.save_pretrained(model_path + "-int4/")

#model = AutoModelForCausalLM.load_low_bit(model_path + "-int4", trust_remote_code=True, optimize_model=True).eval()


input_ids = tokenizer.encode(prompt, return_tensors="pt")
print("finish to load")

model = model.to('xpu')
#model.model.embed_tokens.to('cpu')
#model.transformer.embedding.to('cpu')
input_ids = input_ids.to('xpu')


print("finish to xpu")

#model = BenchmarkWrapper(model)

# with torch.inference_mode():
#     # wamup two times as use ipex
#     for i in range(7):
#         st = time.time()
#         output = model.generate(input_ids, num_beams=1, do_sample=False, max_new_tokens=32)
#         end = time.time()
#         print(f'Inference time: {end-st} s')
#         output_str = tokenizer.decode(output[0], skip_special_tokens=True)
#         print(output_str)

for _ in range(5):
    response_ = ""
    response = ""
    timeFirst = 0
    timeFirstRecord = False
    torch.xpu.synchronize()
    timeStart = time.time()
   # prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=prompt)
    with torch.inference_mode():
        for response, history in stream_chat(model, tokenizer, prompt,32):
           # chatbot[-1] = (input, parse_text(response))
            print(response.replace(response_, ""), end="")  ###  !!!! return 
            response_ = response
            if timeFirstRecord == False:
                torch.xpu.synchronize()
                timeFirst = time.time() - timeStart
                timeFirstRecord = True
           # yield chatbot, history,  "", ""
        timeCost = time.time() - timeStart
    token_count_input = len(tokenizer.tokenize(prompt))
    token_count_output = len(tokenizer.tokenize(response))
    ms_first_token = timeFirst * 1000
    ms_after_token = (timeCost - timeFirst) / (token_count_output - 1+1e-8) * 1000
    print("input: ", prompt)
    print("output: ", response)
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("time cost(s): ", timeCost)
    print("First token latency(ms): ", ms_first_token)
    print("After token latency(ms/token)", ms_after_token)
