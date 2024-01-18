import LLM_mtl_v02

## user text
def text_in(str):
    print("***********prompt:",str)

## model response
def text_out(str):
    print("***********response:",str)

def llm_ms_first_token(str):
    print("***********First Latency:",str)

def llm_ms_after_token(str):
    print("***********After Latency:",str)

def sr_latency(str):
    print("***********SR Latency:",str)

## the wav path of model's response
def wav_path(str):
    print("***********The wav path of model's response:",str)


LLM_mtl_v02.init()

LLM_mtl_v02.predict(audio_input="./hongqiao.wav",text_in=text_in,text_out=text_out,llm_model_select="Baichuan2-7B-Chat",max_length=512,llm_ms_first_token_list=llm_ms_first_token,llm_ms_after_token_list=llm_ms_after_token,sr_latency=sr_latency,save_audio=False,wav_path=wav_path)


#LLM_mtl_v02.predict_user(user_input="你是谁",text_in=text_in,text_out=text_out,llm_model_select="chatglm2-6b",max_length=512,llm_ms_first_token_list=llm_ms_first_token,llm_ms_after_token_list=llm_ms_after_token,save_audio=False,wav_path=wav_path) 


#LLM_mtl_v02.stop_play()