#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  
# 音频文件需要时16k采样，转换为16k方法： ffmpeg -i input.wav -ar 16000 output.wav
# ffmpeg 切分16k音频方法： ffmpeg -i ~/whisper/chinese_5min3s.wav -ar 16000 -ss 00:00:00 -t 10s chinese_10s_16k.wav

# gradio UI loading 太慢 https://github.com/gradio-app/gradio/issues/4332#issuecomment-1563758104
# vim ~/miniconda3/envs/llm/lib/python3.9/site-packages/gradio/themes/utils/fonts.py # 注意需要修改一下虚拟环境的名字
# 大约在第50行的位置，注释return 那一行，并且在前面加pass
# def stylesheet(self) -> str:
#     pass
#     #return f'https://fonts.googleapis.com/css2?family={self.name.replace(" ", "+")}:wght@{";".join(str(weight) for weight in self.weights)}&display=swap'

## 20240116 add history feature
## 20240118 refer to latest 3 round history

import os
import torch
import time
import sys
import traceback
#import argparse
import numpy as np
import scipy

from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
#from transformers import AutoModelForSpeechSeq2Seq

#from transformers import LlamaTokenizer
import intel_extension_for_pytorch as ipex
from transformers import WhisperProcessor
#from transformers import TextStreamer,TextIteratorStreamer
#from colorama import Fore
#import speech_recognition as sr
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
#import gradio as gr
# streamlit as st
#from streamlit_chat import message
import gc
import re
#import mdtex2html
#import intel_extension_for_pytorch as ipex
#import soundfile
#import pyaudio
#import wave
#import TTS
import paddle
from paddlespeech.cli.tts import TTSExecutor
import shutil
import speech_recognition as sr
#from audio_recorder_streamlit import audio_recorder
#from TTS.api import TTS
#from playsound import playsound
#import asyncio
#import edge_tts
from threading import Thread
from multiprocessing import Queue
import pyaudio
import wave
from datetime import datetime
import soundfile
import gradio as gr
import mdtex2html
#from TTS.api import TTS
import langid
from zhconv import convert
from transformers import TextIteratorStreamer
from timer import timer
import onnxruntime as ort
from paddlespeech.t2s.frontend.mix_frontend import MixFrontend
from mix_frontend import MixFrontend
from paddlespeech.t2s.exps.syn_utils import run_frontend

#from openvino.runtime import Core
#from stablediffusionOV import ImageGenerator
#from translate import Translator
import psutil

#from bark import SAMPLE_RATE, generate_audio, preload_models
#from scipy.io.wavfile import write as write_wav
#from FastSpeech2.speech import get_speech

os.environ["USE_XETLA"] = "OFF"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["ENABLE_SDP_FUSION"] = "1"

#device_sd = "gpu.1"
#device_whisper = "xpu"
# you could tune the prompt based on your own model,

#CHATGLM_V2_PROMPT_FORMAT = "问：{prompt}\n\n答："
AQUILA_PROMPT_FORMAT = "Human: {prompt}###Assistant:"
CHATGLM_V2_PROMPT_FORMAT = "问：{prompt}\n答："
INTERNLM_PROMPT_FORMAT = "<|User|>:{prompt}\n<|Bot|>:"

# check if cl_cache exist or not
cache_dir = os.path.expanduser("~/cl_cache")  

if not os.path.exists(cache_dir):
    #create the folder if it does not exist
    os.makedirs(cache_dir)

def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('******************* {} memory used: {} MB'.format(hint, memory))

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)

def get_input_features2(processor, audio_file, device):
    sample_rate = audio_file[0]
    audio = audio_file[1]
    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max #除以65536, 之前demo 是除以32768.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        print("=====rasample audio to 16000Hz")
        audio = resample(audio, sample_rate, 16000)

    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    if device == "xpu":
        input_features = input_features.half().contiguous().to(device)
    #else:
    #    input_features = input_features.contiguous().to(device)
    return input_features

def get_input_features(processor, audio_file, device):
    with sr.AudioFile(audio_file) as source:
        audio = sr.Recognizer().record(source)  # read the entire audio file
    frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
    # audio.ndim == 2:
    #    audio = audio.mean(axis=1)
    if audio.sample_rate != 16000:
        print("=====rasample audio to 16000Hz")
        frame_data = resample(frame_data, audio.sample_rate, 16000)
    input_features = processor(frame_data, sampling_rate=16000, return_tensors="pt").input_features
    if device == "xpu":
       # input_features = input_features.half().contiguous().to(device)
        input_features = input_features.contiguous().to(device)
    return input_features


def load_whisper_model_cpu(model_path, device="cpu"):
   # whisper_model_path = model_path + "/../whisper-small/"
    print("loading whisper---------")
    t0 = time.time()
  
    import whisper as WHisper
    model = WHisper.load_model('small')
    end = time.time()

    if 0:
    # from bigdl.llm import optimize_model
        import librosa
        y, sr = librosa.load("hongqiao.wav")

        # Downsample the audio to 16kHz
        audio = librosa.resample(y,
                            orig_sr=sr,
                            target_sr=16000)

        model = WHisper.load_model('small')

    # model = optimize_model(model)
        st = time.time()
        result = model.transcribe(audio)
        end = time.time()
        print(f'whisper Inference time: {end-st} s')  ## 8s use 13s
        print(result["text"])

    print("loading whisper----------Done, cost time(s): ", end-t0)
    return model 


def load_whisper_model(model_path,sr_model, device):
    whisper_model_path = model_path + sr_model +"-int4/"
    print("loading whisper---------")
    t0 = time.time()

    whisper = AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True, optimize_model=False, tie_word_embeddings=False)
    processor = WhisperProcessor.from_pretrained(whisper_model_path)
   # whisper =  AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, trust_remote_code=True,load_in_4bit=True)
    whisper.config.forced_decoder_ids = None
   # whisper = whisper.half().to(device)
    whisper = whisper.to(device)

    t1 = time.time()
    print("loading whisper----------Done, cost time(s): ", t1-t0)


    return processor, whisper

def load_chatglm2_model(model_path, device):
    chatglm2_model_path = model_path+"-int4"
    print("loading LLM---------",chatglm2_model_path)
    t2 = time.time()
    if model_name_llm == "chatglm2-6b" or model_name_llm == "chatglm3-6b":
        chatglm2_model = AutoModel.load_low_bit(
            chatglm2_model_path,
            trust_remote_code=True,
            optimize_model=True,
            use_cache=True,
            replace_embedding=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            chatglm2_model_path, trust_remote_code=True
        )
    else:
        chatglm2_model = AutoModelForCausalLM.load_low_bit(
            chatglm2_model_path,
            trust_remote_code=True,
            optimize_model=True,
            use_cache=True,
            replace_embedding=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            chatglm2_model_path, trust_remote_code=True
        )
    #chatglm2_model =  AutoModel.load_low_bit(chatglm2_model_path, trust_remote_code=True, optimize_model=False).eval()
   # chatglm2_model = AutoModel.from_pretrained(chatglm2_model_path, trust_remote_code=True, optimize_model=True, load_in_4bit=True).eval()
   # chatglm2_model = AutoModelForCausalLM.load_low_bit(chatglm2_model_path, trust_remote_code=True, optimize_model=True).eval()
    chatglm2_model.to(device)

    ## for MTL iGPU
    #if device == "xpu":
    #    chatglm2_model.transformer.embedding.to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(chatglm2_model_path, trust_remote_code=True)
    torch.xpu.synchronize()
    t3 = time.time()
    print("loading LLM---------Done, cost time(s): ", t3-t2)
    return chatglm2_model, tokenizer


def load_tts_model_paddle(text_in='今天的天气不错啊',audio_out="./audio_cache/output.wav"):
  #  print("loading tts fastspeech2_mix paddle ---------")
  #  tts_executor = TTSExecutor()
    wav_file = tts_executor(
        text=text_in,
        output=audio_out,
        am='fastspeech2_mix',
        am_config=None,
        am_ckpt=None,
        am_stat=None,
        spk_id=174,
        phones_dict=None,
        tones_dict=None,
        speaker_dict=None,
        voc='hifigan_csmsc',
        voc_config=None,
        voc_ckpt=None,
        voc_stat=None,
        lang='mix',
        device='cpu')
    print('***********Wave file has been generated: {}'.format(wav_file))
    #return wav_file

def load_tts_model2(model_path, device):
    print("loading tts fastspeech2_mix---------") 
    show_memory_info("before loading tts ")
    t4 = time.time()
    cpu_threads = 4
    spk_id = 174

    #am = 'fastspeech2_mix'
    phones_dict= model_path + "fastspeech2_mix_onnx_0.2.0/phone_id_map.txt"   
    am_model_path = model_path + "fastspeech2_mix_onnx_0.2.0/fastspeech2_mix.onnx"   
    voc_model_path = model_path + "fastspeech2_mix_onnx_0.2.0/hifigan_csmsc.onnx"
    show_memory_info("before loading tts 1 ")

    tts_frontend = MixFrontend(phone_vocab_path=phones_dict)
    show_memory_info("before loading tts 2 ")
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    show_memory_info("before loading tts 3 ")
    sess_options.intra_op_num_threads = cpu_threads

    am_sess = ort.InferenceSession(am_model_path, providers=providers, sess_options=sess_options)

    voc_sess = ort.InferenceSession(voc_model_path, providers=providers, sess_options=sess_options)
    print("tts fastspeech2_mix load model done! Warmup start----")
    merge_sentences = True

    tts_frontend.get_input_ids(
                "hello, thank you, thank you very much",
                merge_sentences=merge_sentences)
    print("tts fastspeech2_mix load model done! Warmup start  am warmup ----")
    # am warmup
    spk_id = [spk_id]
    for T in [27, 38, 54]:
        am_input_feed = {}
        phone_ids = np.random.randint(1, 266, size=(T, ))
        am_input_feed.update({'text': phone_ids})
        am_input_feed.update({'spk_id': spk_id})
        print(" am warmup 1----")
        am_sess.run(None, input_feed=am_input_feed)
        print(" am warmup 2----")
    print("tts fastspeech2_mix load model done! Warmup start  voc warmup ----")
    # voc warmup
    for T in [227, 308, 544]:
        data = np.random.rand(T, 80).astype("float32")
        voc_sess.run(None, input_feed={"logmel": data})
    print("tts warm up done!")
    t5 = time.time()
    print("loading TTS fastspeech2_mix---------Done, cost time(s): ", t5-t4)

    return tts_frontend, am_sess, voc_sess, spk_id,None
    

def load_tts_model(model_path, device):
    print("loading TTS tacotron2-DDC-GST---------")
    tts_model_path = model_path + "tacotron2-DDC-GST/"
    tts_en_model_path = model_path + "tacotron2-DDC/"
    t4 = time.time()
    tts_model = TTS(model_path=tts_model_path+"model_file.pth", config_path=tts_model_path+"config.json", progress_bar=True, gpu=False) #0.5s
    t5 = time.time()
    print("loading TTS zh---------Done, cost time(s): ", t5-t4)

    print("loading TTS tacotron2-DDC---------")
    t6 = time.time()
    tts_en_model = TTS(model_path=tts_en_model_path+"model_file.pth", config_path=tts_en_model_path+"config.json", progress_bar=True, gpu=False) #0.5s
    t7 = time.time()
    print("loading TTS en---------Done, cost time(s): ", t7-t6)
    return tts_model, tts_en_model

def load_model(model_path, device, model_loaded):
    #try:
    print("******************** Loading Whisper-medium & ChatGLM2 from "+ model_path +" to " + device+" ************************")
    whisper_model_path = model_path+"whisper-medium-int4"
    chatglm2_model_path = model_path+"chatglm2-6b-int4"
    #print("whisper_model_path:", whisper_model_path)
    # device == "xpu" and model_loaded is False:                
    print("loading whisper---------")
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(whisper_model_path)
    whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True, optimize_model=False)
    #whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True)
    whisper.config.forced_decoder_ids = None
    whisper = whisper.half().to(device)
    t1 = time.time()
    print("loading whisper----------Done, cost time(s): ", t1-t0)


   # input_features = get_input_features(processor, "hongqiao.wav", device)#0.09s
    print(" 2 ")
    print("input_features",input_features)
    predicted_ids = whisper.generate(input_features)
    print(" 3 ")
    output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
    print("output_str",output_str)

    print("loading chatglm2---------")
    t2 = time.time()
    chatglm2_model =  AutoModel.load_low_bit(chatglm2_model_path, trust_remote_code=True, optimize_model=False)
    #chatglm2_model = chatglm2_model.half().to('xpu')
    chatglm2_model = chatglm2_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(chatglm2_model_path, trust_remote_code=True)
    t3 = time.time()
    print("loading chatglm2---------Done, cost time(s): ", t3-t2)


    print("=========================total load time(s): ", t5-t4+t3-t2+t1-t0)
    model_loaded = True
 #   return whisper, processor, chatglm2_model, tokenizer, tts_model, tts_en_model, model_loaded
    return whisper, processor, chatglm2_model, tokenizer, None, None, model_loaded

    

def get_prompt(processor, whisper, audio, device):
    with torch.inference_mode():
        print(" 1 ")
        input_features = get_input_features(processor, audio, device)#0.09s
        print("SR device ",device)
        print("input_features",input_features)
        predicted_ids = whisper.generate(input_features)
        print(" 3 ")
        output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
        print("output_str",output_str)
    return output_str


def save_wav(*, wav: list, path: str, sample_rate: int = 22050, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav = np.array(wav)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))


def get_tts2(text, audio_file):
    text = text.replace("\n",",")
    text = text.replace(" ","")
    text = text+"。"
    #tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=True, gpu=False)    tt1 = time.time()           
    tts_model.tts_to_file(text, file_path=audio_file)

def get_tts(text, audio_file):
    text = text.replace("\n",",")
    text = text.replace(" ","")
    text = text+"。"
    #tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=True, gpu=False)    tt1 = time.time()  
    new_text = ""  
    start = 0
    temp = re.split("。|！|？", text)       
    for i in range(len(temp)-1):   
        if len(temp[i]) > 50:
            print("======sentence len > 50, convert!!!", len(temp[i]), temp[i])
            temp[i] = temp[i].replace(",","。")
        new_text += temp[i] + text[start+len(temp[i])]
        start = len(new_text)
    tts_model.tts_to_file(new_text, file_path=audio_file)
    #tts_model.tts_to_file(text, file_path=audio_file)
    #tts_model.tts_to_file(text.replace("\n",",").replace(" ","")+"。", file_path=audio_file)

def get_tts_en(text, audio_file):
    text = text.replace("\n",",")
    tts_en_model.tts_to_file(text, file_path=audio_file)

def get_tts3(text, audio_file):
    fs = 24000
    am_input_feed = {}
    with timer() as t:
        frontend_dict = run_frontend(
            frontend=tts_frontend,
            text=text,
            merge_sentences=False,
            get_tone_ids=False,
            lang="mix")
        phone_ids = frontend_dict['phone_ids']
        flags = 0
        #if len(phone_ids) == 0:
        #    print("len(phone_ids)-----------", len(phone_ids), phone_ids, "\n text: ", text, "\n len text: ", len(text))
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i].numpy()
            am_input_feed.update({'text': part_phone_ids})
            am_input_feed.update({'spk_id': spk_id})
            mel = am_sess.run(output_names=None, input_feed=am_input_feed)
            mel = mel[0]
            wav = voc_sess.run(output_names=None, input_feed={'logmel': mel})
            wav = wav[0]
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = np.concatenate([wav_all, wav])
    #if len(phone_ids) == 0:
    #    return
    #else:
    wav = wav_all
    speed = len(wav) / t.elapse
    rtf = fs / speed
    soundfile.write(audio_file, wav, samplerate=fs)
    print(f"mel: {mel.shape}, wave: {len(wav)}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}.")

def stream_chat(model, tokenizer, prompt, max_new_tokens, history=[],device="xpu"):
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
    #history = []
   # print("history",history)
    output_str = ""
    for stream_output in streamer:
        output_str += stream_output
        yield output_str


def chat(prompt, max_length, top_p, temperature, segment_queue, response_queue, llm_ms_first_token_list, llm_ms_after_token_list,llm_model_name):
    global chatglm2_model, tokenizer,model_name_llm,model_path, history,history_round

    #history = []   ##  0116
    count = 0
    tmp = 2
    all_flag = True
    rest_count = 0

###  0116 add history start    
    if llm_model_name != model_name_llm:
        print("new model--------------restart a new chat")
        history_round = 0
        history = []
        tokenizer = None  ##  0109
    if model_name_llm == "chatglm3-6b":
        max_round = 5
    else:
        max_round = 3

    if tokenizer == None:
        history_round = 0
        history = []
    else:
        if len(tokenizer.tokenize(history)) > 2000 or history_round > max_round:
            print("!!!!!!!!! history len={} > 2k or round={} > {}--------------restart a new chat".format(len(tokenizer.tokenize(history)), history_round, max_round))
          #  history_round = 0  ## 0118
          #  history = []   ## 0118
            history = history.split("\n问：",1)[-1]
        
    if len(history) == 0:
        print("--------------------- new chat ")
        prompt = prompt
        history = prompt         
    else:
        prompt = history + "\n" + prompt
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ round=", history_round)
    print("--------------------------------------history + prompt:\n{}\n---------------------------------------\n".format(prompt))
    history_round += 1

    if llm_model_name != model_name_llm:
        if chatglm2_model != None:
            chatglm2_model.to("cpu")
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            del chatglm2_model
            gc.collect()

        model_name_llm = llm_model_name
        model_full_path = model_path + model_name_llm + "-int4"
        print("******* loading ", model_name_llm)
      #  if load_model_start_callback != None:
      #      load_model_start_callback(model_name_llm)
        stm = time.time()

        if model_name_llm == "chatglm2-6b" or model_name_llm == "chatglm3-6b":
            # chatglm2_model = AutoModel.from_pretrained(model_full_path, trust_remote_code=True, optimize_model=True, load_in_4bit=True).eval()
            chatglm2_model = AutoModel.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )
            chatglm2_model.to("xpu")

        elif model_name_llm == "internlm-chat-20b":
            chatglm2_model = AutoModelForCausalLM.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )
            chatglm2_model.to("xpu")
        else:
            chatglm2_model = AutoModelForCausalLM.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()
            #  model = BenchmarkWrapper(model)).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )
            chatglm2_model.to("xpu")
        print("********** model load time (s)= ", time.time() - stm)
      #  if load_model_finish_callback != None:
      #      load_model_finish_callback(model_name_llm)    
###  0116 add history end       
    
    torch.xpu.synchronize()
    timeStart = time.time()
    timeFirstRecord = False
    with torch.inference_mode():
        for response in stream_chat(chatglm2_model, tokenizer, prompt,max_length, [], "xpu"):
            #print("history******",history)
            response_queue.put(response)
            #response_queue.append(response)
            #message_placeholder.markdown(response)
            # Add a blinking cursor to simulate typing
            if timeFirstRecord == False:
                torch.xpu.synchronize()
                timeFirst = time.time()# - timeStart
                timeFirstRecord = True
                print(f'===============Get first token at {datetime.now()}')

            token_num = len(tokenizer.tokenize(response))
            if  token_num > 300 and all_flag:
                all_flag = False
                #rest_count = count
            elif ("。" in response[rest_count:] or "！" in response[rest_count:] or  \
                "？" in response[rest_count:] or "：" in response[rest_count:]  or \
                    "." in response[rest_count:] or "!" in response[rest_count:] or  \
                "?" in response[rest_count:] or ":" in response[rest_count:] ) and token_num > 40 and all_flag:
                segment = re.split("[。！？：.!?:]", response[rest_count:])
                #print("1-------------segment: ", segment)
                if len(segment) > 1:
                    segment_queue.put(segment[-2])
                    rest_count += len(segment[-2])+1

            elif ("。" in response[rest_count:] or "！" in response[rest_count:] or "，" in response[rest_count:] or \
                "？" in response[rest_count:] or "：" in response[rest_count:] or "," in response[rest_count:] or 
                "." in response[rest_count:] or "!" in response[rest_count:] or \
                "?" in response[rest_count:] or ":" in response[rest_count:] ) and token_num <= 40 and all_flag:
                segment = re.split("[。！？：.!?:，,]", response[rest_count:])
                #print("2-------------segment: ", segment)
                if len(segment) > 1:
                    segment_queue.put(segment[-2])
                    rest_count += len(segment[-2])+1

            count = len(response) 
        timeTotal = time.time()
        llm_time = timeTotal - timeStart
        segment = response[rest_count:]
        #print("=======segment: ", segment)
        segment_queue.put(segment)
        segment_queue.put(None)
        response_queue.put(None)
        #segment_queue.append(None)
        #response_queue.append(None)

    token_count_input = len(tokenizer.tokenize(prompt))
    token_count_output = len(tokenizer.tokenize(response))
    llm_ms_first_token = (timeFirst - timeStart) * 1000
    if token_count_output == 0 or token_count_output == 1:  ## 1220
        llm_ms_after_token = ((timeTotal - timeFirst) * 1000) ### 1220
    elif token_count_output > 1:
        llm_ms_after_token = (
        (timeTotal - timeFirst) / (token_count_output - 1) * 1000)
    
    history = prompt + response
  #  history_return.append(history)
    print("******** max_length history", len(tokenizer.tokenize(history)))
    #yield "", history
    
    print("Prompt: ", prompt)
    print("Response: ", response)    
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("LLM First token latency(ms): ", llm_ms_first_token)
    print("LLM After token latency(ms/token): ", llm_ms_after_token)
    print("LLM time cost(s): ", llm_time)

    llm_ms_first_token_list.append(llm_ms_first_token)
    llm_ms_after_token_list.append(llm_ms_after_token)
    print("-"*50)
    print("\n")


def get_speech(segment_queue, audio_queue, lang):
    while 1:
        text = segment_queue.get()
        #if segment_queue:
            #text = segment_queue.pop(0)
        if text is not None and text != "":
            print("text in get_speech: ", text)
            audio_out = "audio_cache/" + str(datetime.now()).replace(":", "-") + ".wav"
            #asyncio.run(get_tts(text, audio_out))   
            text = text.replace("ChatGLM2-6B", "chat G L M 2 - 6 B")
            for word in re.findall(u"\s[A-Z]+|[A-Z]+\s", text):#往大写字母单词间加空格
                new_word = re.sub(r"(?<=\w)(?=\w)", " ", word)
                text = re.sub(word, new_word, text)
            if lang == "zh": 
                get_tts3(text, audio_out)
              #  load_tts_model_paddle(text, audio_out)
            else:
                #get_tts_en(text, audio_out)
                get_tts3(text, audio_out)
              #  load_tts_model_paddle(text, audio_out)
            data, samplerate = soundfile.read(audio_out) #file does not start with RIFF id
            start = round(np.nonzero(np.array(data))[0].tolist()[0] * 0.95)
           # end = round(np.nonzero(np.array(data))[0].tolist()[-1] * 1.05)  
            end = round(np.nonzero(np.array(data))[0].tolist()[-1] * 1.0)
           # print("data",data)         
           # print("$$$$$$$$$$$$$$$$audio_data: ", len(data), start,end, len(data[start:end]))
            soundfile.write(audio_out, data[start:end], samplerate) # mp3 save to wav
            audio_queue.put(audio_out)
            #audio_queue.append(audio_out)
        else:
            break

def get_speech2(segment_queue, audio_queue):
    while 1:
        text = segment_queue.get()
        #if segment_queue:
            #text = segment_queue.pop(0)
        if text is not None and text != "":
            print("text in get_speech: ", text)
            audio_out = "audio_cache/" + str(datetime.now()).replace(":", "-") + ".wav"
            #asyncio.run(get_tts(text, audio_out))   
            get_tts3(text, audio_out)
            data, samplerate = soundfile.read(audio_out) #file does not start with RIFF id
            start = round(np.nonzero(np.array(data))[0].tolist()[0] * 0.95)
            end = round(np.nonzero(np.array(data))[0].tolist()[-1] * 1.05)           
            #print("$$$$$$$$$$$$$$$$audio_data: ", len(data), start,end, len(data[start:end]))
            soundfile.write(audio_out, data[start:end], samplerate) # mp3 save to wav
            audio_queue.put(audio_out)
            #audio_queue.append(audio_out)
        else:
            break

def play_audio(audio_file = 'sample.wav', CHUNK = 1024): #define play function however you like!
    print(f'~~~~~~~~~~Audio Play starts at {datetime.now()}')
    wf = wave.open(audio_file, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while len(data)!=0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()
    print(f'~~~~~~~~~~~~~~Audio play finishes {datetime.now()}')

def play(audio_queue):
    while 1:
        audio_out = audio_queue.get()
        if audio_out:
            if os.path.exists(audio_out):
                print("~~~playing ", audio_out)
                play_audio(audio_out)
                os.remove(audio_out)  ###############
        else:
            print("~~~~~play done")
            break

def stop_play():
    global  thread1, thread2, thread3
    if thread1:
        print("!!!!!!!!!!!!!! kill thread")
        print(thread1.is_alive())
        print(thread2.is_alive())
        print(thread3.is_alive())
        if thread1.is_alive():
            thread1.kill()
            thread1.join()
        if thread2.is_alive():
            thread2.kill()
            thread2.join()
        if thread3.is_alive():
            thread3.kill()
            thread3.join()
        thread1 = None
        thread2 = None
        thread3 = None
        print("thread1, thread2, thread3 = ",thread1, thread2, thread3)
    try:
        # 删除文件夹及其内容
        shutil.rmtree("./audio_cache")     
        # 新建文件夹
        os.makedirs("./audio_cache")      
        print("****Folder ./audio_cache deleted and recreated.")
    except Exception as e:
        print(f"An error occurred: {e}")

import trace
import threading
class thread_with_trace(threading.Thread): 
  def __init__(self, *args, **keywords): 
    threading.Thread.__init__(self, *args, **keywords) 
    self.killed = False
  def start(self): 
    self.__run_backup = self.run 
    self.run = self.__run       
    threading.Thread.start(self) 

  def __run(self): 
    sys.settrace(self.globaltrace) 
    self.__run_backup() 
    self.run = self.__run_backup 
  def globaltrace(self, frame, event, arg): 
    if event == 'call': 
      return self.localtrace 
    else: 
      return None

  def localtrace(self, frame, event, arg): 
    if self.killed: 
      if event == 'line': 
        raise SystemExit() 
    return self.localtrace 
  def kill(self): 
    self.killed = True


def predict_user(user_function, user_input, chatbot, max_length, top_p, temperature,llm_model_select,play_audio_button):
    if user_function == "语音助手":
        device_select = "xpu"# if device_name == "dGPU" else "cpu"
        #global model_path, device, model_load, whisper, processor, chatglm2_model, tokenizer, model_loaded, tts_model, tts_frontend, am_sess, voc_sess, spk_id
        global thread1, thread2, thread3,tts_model, tts_frontend, am_sess, voc_sess, spk_id#,chatglm2_model, tokenizer,model_name_llm,model_path
        print("\n")
        print("-"*50)

        lang = langid.classify(user_input)[0]
        print("------language: ", lang)
        if lang != "zh" and lang != "en":
            chatbot.append((user_input, "不能识别语言。请使用中文或英文复述一遍"))
            yield chatbot, "", "", ""
            
       
        if llm_model_select == "internlm-chat-7b-8k" or llm_model_select=="internlm-chat-20b":
            prompt = INTERNLM_PROMPT_FORMAT.format(prompt=user_input)
        elif llm_model_select == "AquilaChat2-7B":
            prompt = AQUILA_PROMPT_FORMAT.format(prompt=user_input)
        else:
            prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=user_input)
        #prompt = "What is AI?"
        #prompt = "上海美食指南"
        print("prompt: ", user_input)
       # prompt = convert(prompt, 'zh-cn')
        chatbot.append((user_input, ""))    

        if prompt:
            segment_queue = Queue(50)
            audio_queue = Queue(50)
            response_queue = Queue(50)
            llm_ms_first_token_list = []
            llm_ms_after_token_list = []
            llm_ms_first_token = None
            llm_ms_after_token = None
            
          #  for tid, tobj in threading._active.items():
          #      print("!!!!!!!!!!!!!!!!!!! tid, tobj", tid, tobj)
            print("---start chat----")
            thread1 = thread_with_trace(target=chat, args=(prompt, max_length, top_p, temperature, segment_queue, response_queue, llm_ms_first_token_list, llm_ms_after_token_list,llm_model_select))
            thread1.start()
            if play_audio_button == "True":
                thread2 = thread_with_trace(target=get_speech, args=(segment_queue, audio_queue, lang))
                thread3 = thread_with_trace(target=play, args=(audio_queue,))
                thread2.start()
                thread3.start()

            while 1:  
                response = response_queue.get()
                if response is not None:
                    final_response = response
                #print("-----response: ", response)
                    chatbot[-1] = (user_input, parse_text(response))
                #message_placeholder.markdown(response) 
                if llm_ms_first_token_list:
                    llm_ms_first_token = llm_ms_first_token_list.pop(0)
                    llm_ms_first_token = str(round(llm_ms_first_token, 2)) + " ms"
                    print("---------llm_ms_first_token:", llm_ms_first_token)
                if llm_ms_after_token_list:
                    llm_ms_after_token = llm_ms_after_token_list.pop(0)
                    llm_ms_after_token = str(round(llm_ms_after_token, 2)) + " ms/token"
                if response is None:
                    #message_placeholder.markdown(final_response)    
                    chatbot[-1] = (user_input, parse_text(final_response))                        
                    #st.session_state.messages.append({"role": "assistant", "content": response})                        
                    break          

                yield chatbot, None, llm_ms_first_token, llm_ms_after_token

            yield chatbot, None, llm_ms_first_token, llm_ms_after_token

def predict(user_function, audio_input, chatbot, max_length, top_p, temperature,sr_model,device_name,llm_model_select,play_audio_button):  
    if user_function == "语音助手":
        device_select = "xpu"# if device_name == "dGPU" else "cpu"
        
        #global model_path, device, model_load, whisper, processor, chatglm2_model, tokenizer, model_loaded, tts_model, tts_frontend, am_sess, voc_sess, spk_id
        global thread1, thread2, thread3,sr_model_now,processor, whisper,device_sr,device_select_sr,tts_model, tts_frontend, am_sess, voc_sess, spk_id

        if sr_model_now != sr_model or device_sr != device_name:
            print("Change SR model from to ",sr_model_now, sr_model)
            print("Change SR inference device from to ",device_sr, device_name)
            sr_model_now = sr_model
            device_sr = device_name
            if device_sr == "CPU":    
                device_select_sr = "cpu"    
                processor, whisper = load_whisper_model(model_path,sr_model_now ,device_select_sr)
            elif device_sr == "iGPU":
                device_select_sr = "xpu"
                processor, whisper = load_whisper_model(model_path,sr_model_now ,device_select_sr)

        print("\n")
        print("-"*50)

       # audio_input = "hongqiao.wav"

        print("\n ******** 1")
        torch.xpu.synchronize()
        
        t0 = time.time()
        prompt_in = get_prompt(processor, whisper, audio_input, device_select_sr)
        print("\n ******** 2")
        torch.xpu.synchronize()
        t1 = time.time()
        sr_latency = (t1 - t0) * 1000
        print("SR time cost(ms): ", sr_latency)
        sr_latency = str(round(sr_latency, 2)) + " ms"
  

        #prompt_in = "What is AI?"
        #prompt_in = "上海美食指南"
        print("prompt: ", prompt_in)
        
        prompt_in = convert(prompt_in, 'zh-cn')
        
       # predict_user(user_function, prompt_in, chatbot, max_length, top_p, temperature,llm_model_select)
        #yield "", "", sr_latency, "", ""
        
        lang = langid.classify(prompt_in)[0]
        print("------language: ", lang)
        if lang != "zh" and lang != "en":
            chatbot.append((prompt_in, "不能识别语音。请使用中文或英文复述一遍"))
            yield chatbot, "", "", ""

        if llm_model_select == "internlm-chat-7b-8k":
            prompt = INTERNLM_PROMPT_FORMAT.format(prompt=prompt_in)
        elif llm_model_select == "AquilaChat2-7B":
            prompt = AQUILA_PROMPT_FORMAT.format(prompt=prompt_in)
        else:
            prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=prompt_in)

       # prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=prompt_in)
        chatbot.append((prompt_in, ""))    

        if prompt:
            segment_queue = Queue(50)
            audio_queue = Queue(50)
            response_queue = Queue(50)
            llm_ms_first_token_list = []
            llm_ms_after_token_list = []
            llm_ms_first_token = None
            llm_ms_after_token = None

            for tid, tobj in threading._active.items():
                print("!!!!!!!!!!!!!!!!!!! tid, tobj", tid, tobj)
            print("---start chat----")

            thread1 = thread_with_trace(target=chat, args=(prompt, max_length, top_p, temperature, segment_queue, response_queue, llm_ms_first_token_list, llm_ms_after_token_list,llm_model_select))
            thread1.start()
            if play_audio_button == "True":
                thread2 = thread_with_trace(target=get_speech, args=(segment_queue, audio_queue, lang))
                thread3 = thread_with_trace(target=play, args=(audio_queue,))          
                thread2.start()
                thread3.start()
           # for tid, tobj in threading._active.items():
           #     print("!!!!!!!!!!!!!!!!!!! tid, tobj", tid, tobj)

            while 1:  
                response = response_queue.get()
                if response is not None:
                    final_response = response
                #print("-----response: ", response)
                    chatbot[-1] = (prompt_in, parse_text(response))
                #message_placeholder.markdown(response) 
                if llm_ms_first_token_list:
                    llm_ms_first_token = llm_ms_first_token_list.pop(0)
                    llm_ms_first_token = str(round(llm_ms_first_token, 2)) + " ms"
                    print("---------llm_ms_first_token:", llm_ms_first_token)
                if llm_ms_after_token_list:
                    llm_ms_after_token = llm_ms_after_token_list.pop(0)
                    llm_ms_after_token = str(round(llm_ms_after_token, 2)) + " ms/token"
                
                if response is None:
                    #message_placeholder.markdown(final_response)    
                    chatbot[-1] = (prompt_in, parse_text(final_response))                        
                    #st.session_state.messages.append({"role": "assistant", "content": response})                        
                    break          

                yield chatbot, sr_latency, llm_ms_first_token, llm_ms_after_token

            yield chatbot, sr_latency, llm_ms_first_token, llm_ms_after_token

  

def choose_function(user_function):
    if user_function == "语音助手":
        return [gr.update(value="", visible=True), gr.update(visible=False), gr.update(visible=False),  gr.update(visible=False)]
    else:
        return [gr.update(value="", visible=False), gr.update(visible=True),  gr.update(visible=True),  gr.update(visible=False)]

def reset_user_input():
    return gr.update(value='')

def reset_state():
    global history,history_round
    history = []
    history_round = 0
    return [],gr.update(value=''),gr.update(value=''),gr.update(value='')


if __name__ == "__main__":
    global model_path, device, model_load, whisper, processor, chatglm2_model, tokenizer, model_loaded, pipe,  tts_executor, thread1, thread2, thread3,sr_model_now,device_sr,model_name_llm,tts_frontend, am_sess, voc_sess, spk_id,tts_en_model
    model_path = "./models/"
  #  device_list = ["iGPU", "CPU"]
    #device_list = ["iGPU"]
    history = []
    history_round = 0
    device = "None"
    device_sr = "iGPU"
    model_load = False
    thread1 = None
    thread2 = None
    thread3 = None
    sr_model_now  = "whisper-small"
    model_name_llm = "Baichuan2-7B-Chat"

    device_select = "xpu"# if device_name == "dGPU" else  "cpu"
    device_select_sr = "xpu"
    model_loaded = False
    #whisper, processor, chatglm2_model, tokenizer, tts_model, tts_en_model, model_loaded = load_model(model_path, device_select, model_loaded)
    print("loading tts fastspeech2_mix paddle ---------")
    tts_frontend, am_sess, voc_sess, spk_id, tts_en_model = load_tts_model2(model_path, device_select)
    processor, whisper = load_whisper_model(model_path,sr_model_now ,device_select_sr)
    
    chatglm2_model, tokenizer = load_chatglm2_model(model_path + model_name_llm, device_select)
    
    if not os.path.exists("audio_cache"):
        os.mkdir("audio_cache")

    """Override Chatbot.postprocess"""
    gr.Chatbot.postprocess = postprocess
  
    with gr.Blocks(theme=gr.themes.Base.load("theme3.json"), css=".gradio-container {display:flex;flex-wrap: wrap;} .radio-group .wrap {display: grid !important;grid-template-columns: 1fr;} footer {visibility: hidden}") as demo:
    #with gr.Blocks(css=".gradio-container {display:flex;flex-wrap: wrap;} footer {visibility: hidden}") as demo:
        gr.HTML("""<h1 align="center">英特尔大语言模型应用</h1>""")
        with gr.Row():           
            with gr.Column(scale=2, visible=True):
                with gr.Column():
                    sr_model = gr.Dropdown(["whisper-medium","whisper-small"],label="语音识别模型", value="whisper-small")
                    device_name = gr.Dropdown(["DGPU", "CPU"],value="DGPU",label="选择语音识别推理设备", interactive=True)
                    
                with gr.Column(scale=1, visible=True): # 配置是否显示控制面板 
                    llm_model_select = gr.Dropdown(["Baichuan2-7B-Chat","Baichuan2-13B-Chat"],label="对话模型", value="Baichuan2-7B-Chat")   
                    device_name_llm = gr.Textbox(label="对话模型推理设备", value="DGPU")                   
                    max_length = gr.Slider(1, 2048, value=512, step=1.0, label="最大输出长度", interactive=True)                       
                    temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True, visible=False)
                    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True, visible=False)
                    play_audio_button = gr.Dropdown(["True","False"],label="语音播放对话", value="True") 
                with gr.Column():
                    sr_latency = gr.Textbox(label="语音识别耗时", visible=True)
                    llm_f_latency = gr.Textbox(label="First Latency", visible=True)
                    llm_a_latency = gr.Textbox(label="After Latency", visible=True)

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(show_label=False, scale=1, avatar_images=["user.png", "AI.png"], height=650)            
                prompt_img = gr.Textbox(label = "提示词", show_label=True, visible=False)
                promptTranslated = gr.Textbox(label = "English",show_label=True, visible=False)
                img = gr.Image(scale=1,label="Image for txt2img", show_label=False, type="pil", image_mode="RGBA", height=540, visible=False)
                #audio_output = gr.Audio(type="filepath", label="Generated Audio")
                with gr.Row():
                    with gr.Column(scale=8):
                        audio_input = gr.Audio(source="microphone", type="filepath", interactive=True)
                    with gr.Column(scale=2, visible=False):

                        user_function = gr.Radio(["语音助手"], value="语音助手", label="", elem_classes="radio-group",interactive=True)
                user_input = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=5,interactive=True)
                with gr.Row():
                    submitBtn = gr.Button("提交文字", variant="primary",interactive=True)
                    emptyBtn = gr.Button("清除历史记录",interactive=True)
                    stopBtn = gr.Button("停止播放",interactive=True)


        history_gr = gr.State([])
        user_function.change(choose_function, [user_function], [chatbot, img, prompt_img, promptTranslated])

        audio_input.stop_recording(predict, [user_function, audio_input, chatbot, max_length, top_p, temperature, sr_model,device_name,llm_model_select,play_audio_button],
                            [chatbot, sr_latency, llm_f_latency, llm_a_latency], show_progress=True)

        submitBtn.click(predict_user, [user_function, user_input, chatbot, max_length, top_p, temperature,llm_model_select,play_audio_button],
                        [chatbot, sr_latency, llm_f_latency, llm_a_latency], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot,sr_latency, llm_f_latency, llm_a_latency], show_progress=True)

        stopBtn.click(stop_play, [],[])

    
    demo.queue().launch(share=False, inbrowser=True)

    


            

