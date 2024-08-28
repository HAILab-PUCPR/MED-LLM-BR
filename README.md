# MED-LLM-BR: A Large Language Model for Clinical Text Generation in Portuguese

## Introduction:
MED-LLM-BR is a project focused on fine-tuning large language models specifically for generating clinical notes in Portuguese. This project leverages LLama and Mistral as base models, adapting them through fine-tuning techniques to enhance their performance in clinical text generation tasks.

To optimize resource utilization during the fine-tuning process, we employed Low-Rank Adaptation (LoRA). This approach enables effective model adaptation with significantly reduced computational and memory requirements, making the fine-tuning process more efficient without compromising the quality of the generated clinical text.
Model Description


## Models
LLama: LLama is a state-of-the-art language model known for its scalability and efficiency in handling diverse natural language processing tasks. In this project, LLama serves as one of the base models for fine-tuning, aimed at adapting it to the specific requirements of clinical text generation in Portuguese.

Mistral: Mistral is another advanced language model designed to enhance performance in various text generation applications. By incorporating Mistral into the fine-tuning pipeline, MED-LLM-BR seeks to combine the strengths of multiple models to achieve superior results in generating accurate and contextually relevant clinical notes.

-------------------------------------------------------------------------------------------------------------------------------------------------------


## How to use or models with HuggingFace

Link: https://huggingface.co/pucpr/gpt2-bio-pt

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
login()


model_id = "pucpr-br/Clinical-BR-LlaMA-2-7B" or "pucpr-br/Clinical-BR-Mistral-7B-v0.2"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = ""Paciente admitido com angina instável, progredindo para infarto agudo do miocárdio (IAM) inferior no primeiro dia de internação; encaminhado para unidade de hemodinâmica, onde foi feita angioplastia com implante de stent na ponte d"	"
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
print(inputs)

outputs = model.generate(**inputs, max_new_tokens=90)

Resposta
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
---------------------------------------------------------------------------------------------------------------------------------------------------------

Resultado:
"Paciente admitido com angina instável, progredindo para infarto agudo do miocárdio (IAM) inferior no primeiro dia de internação; encaminhado para unidade de hemodinâmica, onde foi feita angioplastia com implante de stent na ponte dorsal da arteria coronária direita. Paciente apresentou IAM em 1º dia de internação, com desvio de 60% no segmento proximal, com angiografia de alta qualidade. Anteriormente, paciente havia apresentado episódios de angina instável, em 2003, com angiografia com desvio de 70% no segment..."


# Citation:
@INPROCEEDINGS{,
  author={João Gabriel de Souza Pinto,Andrey Rodrigues de Freitas, Anderson Carlos Gomes Martins,Caroline Midori Rozza Sawazaki, Caroline Vidal, Lucas Emanuel Silva e},
  booktitle={}, 
  title={Developing Resource-Efficient Clinical LLMs for Brazilian Portuguese}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}
}
