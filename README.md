<div align="center">
  <img src="https://github.com/user-attachments/assets/638ba60d-606b-4b5d-a549-abd411f9886e" width="300"/>
</div>



----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MED-LLM-BR: Medical Large Language Models for Brazilian Portuguese
MED-LLM-BR is a collaborative project between [HAILab](https://github.com/HAILab-PUCPR) and [Comsentimento](https://www.comsentimento.com.br/), which aims to develop multiple medical LLMs for Portuguese language, including base models and task-specific models, with different sizes. 

## Contributions:

### Developing Resource-Efficient Clinical LLMs for Brazilian Portuguese
This project leverages LLama and Mistral as base models, adapting them through fine-tuning techniques to enhance their performance in clinical text generation tasks.

To optimize resource utilization during the fine-tuning process, we employed Low-Rank Adaptation (LoRA). This approach enables effective model adaptation with significantly reduced computational and memory requirements, making the fine-tuning process more efficient without compromising the quality of the generated clinical text.

#### Model Description
LLama: LLama is a state-of-the-art language model known for its scalability and efficiency in handling diverse natural language processing tasks. In this project, LLama serves as one of the base models for fine-tuning, aimed at adapting it to the specific requirements of clinical text generation in Portuguese.

Mistral: Mistral is another advanced language model designed to enhance performance in various text generation applications. By incorporating Mistral into the fine-tuning pipeline, MED-LLM-BR seeks to combine the strengths of multiple models to achieve superior results in generating accurate and contextually relevant clinical notes.

#### How to use the models with HuggingFace

Link model 1: [Clinical-BR-LlaMA-2-7B](https://huggingface.co/pucpr-br/Clinical-BR-LlaMA-2-7B)

Link model 2: [Clinical-BR-Mistral-7B-v0.2](https://huggingface.co/pucpr-br/Clinical-BR-Mistral-7B-v0.2)

~~~bash
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from huggingface_hub import login
login()

model_id = "pucpr-br/Clinical-BR-LlaMA-2-7B" or "pucpr-br/Clinical-BR-Mistral-7B-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model     = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
prompt = "Paciente admitido com angina instável, progredindo para infarto agudo do miocárdio (IAM) inferior no primeiro dia de internação; encaminhado para unidade de hemodinâmica, onde foi feita angioplastia com implante de stent na ponte d "	
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
outputs = model.generate(**inputs, max_new_tokens=90)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
~~~

#### How to use the models with ollama

Model Clinical-BR-LlaMA-2-7B:

~~~bash
$ ollama run cabelo/clinical-br-llama-2-7b
~~~

Model Clinical-BR-Mistral-7B-v0.2:

~~~bash
$ ollama run cabelo/clinical-br-mistral-7b-0.2
~~~
#### How to use the OpenVINO models with HuggingFace

Link model FP16 : [Clinical-BR-LlaMA-2-7B-fp16-ov](https://huggingface.co/cabelo/Clinical-BR-LlaMA-2-7B-fp16-ov)

Link model Int8  : [Clinical-BR-LlaMA-2-7B-int8-ov](https://huggingface.co/cabelo/Clinical-BR-LlaMA-2-7B-int8-ov) 

Source and more information in [MED-LLM-BR-openvino](https://github.com/cabelo/MED-LLM-BR-openvino)

#### Provisional Citation:
```
@inproceedings{pinto2024clinicalLLMs,
  title        = {Developing Resource-Efficient Clinical LLMs for Brazilian Portuguese},
  author       = {João Gabriel de Souza Pinto and Andrey Rodrigues de Freitas and Anderson Carlos Gomes Martins and Caroline Midori Rozza Sawazaki and Caroline Vidal and Lucas Emanuel Silva e Oliveira},
  booktitle    = {Proceedings of the 34th Brazilian Conference on Intelligent Systems (BRACIS)},
  year         = {2024},
  note         = {In press},
}
```
