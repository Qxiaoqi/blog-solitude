---
title: 大模型 Fine-tuning 实践
date: 2024-08-19 17:57:00
toc: true
recommend: true
categories:
- 大模型
tags: 
- AI
- 大模型
cover: https://file-1305436646.file.myqcloud.com/blog/banner/2024-08-19.webp
---

# 大模型 Fine-tuning 实践

前面两篇文章整理了学习笔记，对大模型应该有了一个初步认知。这篇文章会从代码角度，尝试一下微调。看完本篇文章之后，应该能自己训练一个简单的类 GPT demo 模型。

## 一、准备工作

### 1. 训练数据集

比如我下面的数据是我自己开发的 [Capybara 翻译软件](https://capybara-translate.cn/docs/intro) 文档的 txt 格式数据集其中之一，这里只是一个简单的例子，实际上你可以使用任何文本数据集。你可以在这里找到更多数据集 [huggingface dataset hub](https://huggingface.co/datasets)

```txt
Capybara 翻译软件的主要特性:

1. 多平台翻译比对：支持多种翻译平台间的翻译结果对比，帮助用户选择最佳翻译。
2. 强大的 OCR 识别：准确提取桌面展示图片的文字内容。
3. 跨平台支持：兼容 windows 和 macOS 系统，提供更好的使用体验。
4. 性能极佳：采用 Rust 开发的桌面应用，性能优越。
5. AI 驱动：利用人工智能技术提升翻译质量和用户体验。
6. GPT 翻译按需计费：根据使用 token 计费，避免长期包月费用，用多少付多少。其他内置翻译不计费。
```

### 2. 开源模型

- Qwen/Qwen2-0.5B
- Qwen/Qwen2-0.5B-Instruct

我们这里选用通义千问来做微调。你可以在 huggingface 去找到这两个模型，当然你也可以使用其他模型。选择 0.5B 参数量是因为模型相对较小，训练速度更快。

## 二、数据准备

- 数据转换
- Tokenization

数据准备分为两部分，一部分是数据转换，另一部分是 Tokenization。

### 1. 数据转换

```python
import torch
import datasets
import os
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## data preprocess
import json
# 1. write a python function to transform raw text to json format
def transform_files_in_directory(directory_path, output_file_path):
    data_list = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if not os.path.isfile(file_path):
            continue
        logger.info(f'reading file {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            data_list.append({'text': text})
    logger.info(f"output the processed data to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)

output_file_path = './data/train/pretrain.json'
input_directory_path = './data/raw'
transform_files_in_directory(input_directory_path, output_file_path)
```

上面的代码将 txt 格式的数据处理为 JSON 格式，方便后续处理。转换结果如下：

```json
[
  {
    "text": "Capybara 是一款多平台翻译软件，支持 OCR 识别翻译，并且内置 GPT 翻译服务，努力成为您最好的办公助手!"
  },
  {
    "text": "Capybara 翻译软件计费规则如下\n\n计费按照 官方的调用费用 * 1.3计算\n\n官方调用费用如下：\n\n我们采用的是 GPT-3.5-Turbo-0613 版本，输入每 1000 token 为 $0.0015，输出每 1000 token 为 $0.002，我们会按照实时汇率计算人民币。\n\n例子\n1 RMB = 10 点数\n\n100 字英语短文：\n\n我们以一个例子来说明大概的费用情况，比如一个 100 字英文短文，我们用 GPT 翻译成中文，大概需要 0.07 点数，也就是 0.007 RMB。"
  },
  {
    "text": "Capybara 翻译软件的主要特性:\n\n1. 多平台翻译比对：支持多种翻译平台间的翻译结果对比，帮助用户选择最佳翻译。\n2. 强大的 OCR 识别：准确提取桌面展示图片的文字内容。\n3. 跨平台支持：兼容 windows 和 macOS 系统，提供更好的使用体验。\n4. 性能极佳：采用 Rust 开发的桌面应用，性能优越。\n5. AI 驱动：利用人工智能技术提升翻译质量和用户体验。\n6. GPT 翻译按需计费：根据使用 token 计费，避免长期包月费用，用多少付多少。其他内置翻译不计费。"
  },
  {
    "text": "Capybara 翻译软件如何使用\n\n使用说明:\n\n1. 主界面\n安装完成后，打开软件，你会看到如下界面，默认没登陆情况无法使用翻译功能，你需要先登录。\n\n2. 登陆\nCapybara 翻译平台不会存储用户的隐私信息，密码会经过加密后存储，安全可靠。\n点击左上角的【未登陆】或者左下角的【设置】按钮\n\n输入邮箱和密码，点击【登陆/注册】按钮，首次登陆会给邮箱发送验证码，输入验证码即可登陆。\n\n邮箱验证码是类似下图的邮件，里面会携带六位验证码：\n\n3. 个人中心\n1 RMB = 10 点数\n\n个人中心可以查看自己当天的调用次数，以及当天消耗的点数，还有总的剩余点数。\n\n4. 使用翻译\n\n这是一段 GPT 随机生成的 100 字英语短文，我们用它来测试 OCR 翻译功能。\n\n主界面上点击【截图 OCR 翻译】，或者使用快捷键 Ctrl + S，即可调出 OCR 识别选取。"
  }
]
```

### 2. Tokenization

Tokenization 可以在 [the-tokenizer-playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) 尝试。

```python
# load the pretraining dataset and do tokenization
# the model only accepts the tokenized text as input, each token is represented by a unique integer id
from datasets import load_dataset
data_path = "data/train/pretrain.json"
train_data = load_dataset("json", data_files=data_path, split='train')
tokenizer = AutoTokenizer.from_pretrained('models/Qwen2-0.5B')

encoded_texts = tokenizer.encode(train_data[0]['text'], add_special_tokens=False)
print("Encoded token id:", encoded_texts)  # print the first document's encoded token ids

from typing import Dict, List

def process_func(examples: Dict[str, List]):
    max_token = 1024
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False, max_length=max_token, truncation=True)  # tokenize the texts
    input_ids_list = encoded_texts['input_ids']

    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))
    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list
    }

print(train_data.column_names)
train_data = train_data.map(
    process_func,
    batched=True,
    num_proc=1,
    remove_columns=train_data.column_names,
    desc='Running tokenizer on train_set: '
)
print(train_data)
```

比如第一条数据，经过 Tokenization 之后的结果如下：

```
Encoded token id: [34, 9667, 24095, 54851, 104794, 42140, 100133, 105395, 103951, 3837, 100143, 80577, 220, 102450, 105395, 90395, 100136, 115654, 479, 2828, 10236, 123, 119, 102610, 47874, 3837, 101066, 99787, 87026, 102235, 100757, 110498, 0]
```

## 三、模型微调

- 加载模型
- 设置模型参数
- 运行训练过程
- 推理 & 验证

### 1. 加载模型

```python
# load a pretrained model with huggingface/transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "./models/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
```

打印出模型信息：

```python
# we print the model parameters

def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("----------------------------")
    total_params = 0
    emb_size = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        if 'embed_tokens' in name or 'lm_head' in name:
            emb_size += param_count
        total_params += param_count
        print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
    print("----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")
    print(f"Embedding & Un-embedding layer parameters percentage: {emb_size/total_params * 100: 0.2f}%")

print_model_parameters(model)

# this model is a transformer, the hidden size is 896, the number of layers is 24
```

#### 1.1 测试模型

这里先测试一下通义千问模型，看看推理结果：

```python
# test the loaded model

def inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str = "Capybara",
    max_new_tokens: int = 128
):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    # print(outputs)
    print(generated_text)

inference(model, tokenizer, input_text="Capybara 翻译软件的主要特性:")
```

模型输出结果如下：

```
Capybara 翻译软件的主要特性: 1. 2000 多种编程语言支持; 2. 丰富的资源和工具，如： Python, C, C++, Java, JavaScript 等编程语言，提供详细的官方教程; 3. 超过 300 万条示例; 4. 丰富的工具箱，包括编译器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、调试器、
```

可以看到基本是胡言乱语了，因为数据集里面并没有 Capybara 的信息。下一步，就是微调这个模型，让他能够生成我们想要的文本。

### 2. 设置模型参数

设置模型，注意一下 `learning_rate` 参数。

> 学习率 是一个超参数，用于控制在每次更新模型参数时，梯度下降算法的步伐大小。简单来说，它决定了每次调整模型参数的幅度。

设置的比较小，可能得不到想要的训练结果，具体数值可以自行尝试。

```python
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# after the data & model loading, we will do the training next, first we should set the training arguments, and initialize the trainer

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir='saves',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    eval_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    lr_scheduler_type='cosine',
    bf16=False,
    fp16=False,
    logging_steps=1,
    report_to=None,
    num_train_epochs=5,
    save_steps=1000,
    save_total_limit=1,
    warmup_ratio=0.1,
    use_cpu=True,  # we train the model on cpu
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

### 3. 运行训练过程

```python
# start the training process, just 1 line of code

trainer.train()
print("training finished")
# save the final model
trainer.save_model('saves/Qwen2-0.5B-pt')
```

训练结果：

```
{'loss': 0.2601, 'grad_norm': 9.022148132324219, 'learning_rate': 3e-05, 'epoch': 1.0}
{'loss': 0.2601, 'grad_norm': 9.022154808044434, 'learning_rate': 2.5606601717798212e-05, 'epoch': 2.0}
{'loss': 0.2355, 'grad_norm': 4.477291584014893, 'learning_rate': 1.5e-05, 'epoch': 3.0}
{'loss': 0.0913, 'grad_norm': 3.026254653930664, 'learning_rate': 4.393398282201788e-06, 'epoch': 4.0}
{'loss': 0.0386, 'grad_norm': 2.367913007736206, 'learning_rate': 0.0, 'epoch': 5.0}
{'train_runtime': 189.7726, 'train_samples_per_second': 0.105, 'train_steps_per_second': 0.026, 'train_loss': 0.1771329365670681, 'epoch': 5.0}
training finished
```

可以尝试看一下这个结果，loss 数值越来越小，说明模型在训练过程中逐渐收敛。epoch 为 5，表示训练了 5 轮。learning_rate 逐渐减小，这是因为我们设置了 `lr_scheduler_type='cosine'`，这个参数会让学习率逐渐减小。

### 4. 推理 & 验证

```python
pretrained_model = AutoModelForCausalLM.from_pretrained('saves/Qwen2-0.5B-pt')
inference(pretrained_model, tokenizer, input_text="Capybara 翻译软件的主要特性:", max_new_tokens=128)
```

输出结果：

```
Capybara 翻译软件的主要特性: 

1. 多平台翻译比对
2. 支持 OCR 识别翻译，准确翻译各类文本
3. 强大的 GPT 翻译服务，提供实时翻译效果
4. 性能极佳，采用 Rust 开发的翻译系统，性能优越
5. 多平台兼容，支持多种翻译平台间的翻译结果共享
6. 支持 GPT 翻译服务，提供更加便捷的翻译体验
7. 性能极佳，采用 Rust 开发的翻译系统，性能优越

总结一下，这款翻译软件主要面向办公场景，
```

可以看到，虽然有少许的差异，但是总体还是很精准的。最开始设置的 `learning_rate` 比较小，没有得到想到的结果，将其调大之后，以及 `epoch` 调大，明显比最开始的效果好很多。

#### 4.1 问答

这里用一个问句来测试是否能得到我们想要的结果。

```python
input_text2 = "你帮我回答一下 Capybara 翻译软件的主要特性有什么?"
inference(pretrained_model, tokenizer, input_text=input_text2, max_new_tokens=256)
```

输出结果：

```
你帮我回答一下 Capybara 翻译软件的主要特性有什么? 1. 多平台翻译比对：支持多种翻译平台间的翻译结果对比，帮助用户选择最佳翻译。
2. 智能翻译比对：使用人工智能技术，帮助用户快速准确翻译多语种文章。
3. 跨平台支持：兼容多种操作系统，提供更好的使用体验。
4. 性能极佳：采用 Rust 开发的翻译引擎，性能优越。
5. 性能极佳：采用 Rust 开发的翻译引擎，性能优越。
6. 多平台支持：兼容多种翻译平台间的翻译结果对比，帮助用户选择最佳翻译。
7. GPT 随机生成：提供 GPT 随机生成翻译服务，提高翻译质量和用户体验。
8. AI 驱动：利用人工智能技术提升翻译质量和用户体验。
9. 翻译准确率高：采用 GPT 翻译技术，翻译准确率高。
10. GPT 随机生成：提供 GPT 随机生成翻译服务，提高翻译质量和用户体验。
```

可以看到，整体答案还是符合的，但是有一点，模型把我们的问题也作为回答的一部分了。可以得出，模型学习了知识，但无法按照我们的指令进行响应，预期是能按照类似 GPT 的回答方式，一问一答。因此我们需要指令调优，即 SFT。

## 四、指令调优

指令调优是指在微调模型的过程中，通过指令的方式来指导模型生成更加符合预期的结果。这里我们使用的是通义千问的 Qwen2-0.5B-Instruct 模型。

- 数据格式化
- 训练
- 推理

### 1. 数据格式化

直接让 GPT 帮我们格式化需要的指令数据，后面用这个去训练：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Capybara 是什么？"
      },
      {
        "role": "assistant",
        "content": "Capybara 是一款多平台翻译软件，支持 OCR 识别翻译，并且内置 GPT 翻译服务，努力成为您最好的办公助手!"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "Capybara 翻译软件的主要特性是什么？"
      },
      {
        "role": "assistant",
        "content": "Capybara 翻译软件的主要特性:\n\n1. 多平台翻译比对：支持多种翻译平台间的翻译结果对比，帮助用户选择最佳翻译。\n2. 强大的 OCR 识别：准确提取桌面展示图片的文字内容。\n3. 跨平台支持：兼容 windows 和 macOS 系统，提供更好的使用体验。\n4. 性能极佳：采用 Rust 开发的桌面应用，性能优越。\n5. AI 驱动：利用人工智能技术提升翻译质量和用户体验。\n6. GPT 翻译按需计费：根据使用 token 计费，避免长期包月费用，用多少付多少。其他内置翻译不计费。"
      }
    ]
  },
  // ...
]
```

### 2. 设置参数

```python
# so we have done the pretraing process, the model just can just predict the next token based on the previous tokens
# but how to make the model following my instructions like chatGPT do?
# inorder to obtain the capability of instruction following,
# now we will do the instruction fine-tuning process, we will use the same model, but we will use the fine-tuning dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


training_args = SFTConfig(
    output_dir='saves',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    eval_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,  # you can tune the learning rate and see what's the impact
    lr_scheduler_type='cosine',
    bf16=False,
    fp16=False,
    logging_steps=1,
    report_to=None,
    num_train_epochs=5,
    save_steps=1000,
    save_total_limit=1,
    warmup_ratio=0.1,
    use_cpu=True,
    seed=12345
)
checkpoint_path = "models/Qwen2-0.5B-Instruct"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
print(tokenizer.padding_side)
##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

sft_data = load_dataset('json', data_files='data/train/sft_train.json', split='train')
```

### 3. 处理数据

```python
from trl import SFTConfig, SFTTrainer
from trl import DataCollatorForCompletionOnlyLM

column_names = list(sft_data.features)
processed_dataset = sft_data.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=1,
    remove_columns=column_names,
    desc="Applying chat template",
)
response_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
sft_trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    max_seq_length = 1024,
    dataset_text_field="text",
    tokenizer=tokenizer,
    data_collator=collator,
)
# have a look at the first formated sample
processed_dataset[0]['text']
```

格式化样本：

```
'<|im_start|>system\n<|im_end|>\n<|im_start|>user\nCapybara 是什么？<|im_end|>\n<|im_start|>assistant\nCapybara 是一款多平台翻译软件，支持 OCR 识别翻译，并且内置 GPT 翻译服务，努力成为您最好的办公助手!<|im_end|>\n'
```

上面这条数据即为格式化后的数据，到这里，指令已经被正确的格式化了。

### 4. 训练

```python
train_result = sft_trainer.train()
metrics = train_result.metrics
sft_trainer.log_metrics("train", metrics)
sft_trainer.save_metrics("train", metrics)
sft_trainer.save_model('saves/Qwen2-0.5B-sft2')
```

训练结果：

```
{'loss': 2.796, 'grad_norm': 39.54423904418945, 'learning_rate': 1.5e-05, 'epoch': 0.25}
{'loss': 2.4956, 'grad_norm': 54.94034957885742, 'learning_rate': 3e-05, 'epoch': 0.5}
{'loss': 2.8228, 'grad_norm': 27.41976547241211, 'learning_rate': 2.977211629518312e-05, 'epoch': 0.75}
{'loss': 2.609, 'grad_norm': 49.70899200439453, 'learning_rate': 2.9095389311788626e-05, 'epoch': 1.0}
{'loss': 1.3217, 'grad_norm': 21.66421890258789, 'learning_rate': 2.7990381056766583e-05, 'epoch': 1.25}
{'loss': 0.5918, 'grad_norm': 24.359455108642578, 'learning_rate': 2.649066664678467e-05, 'epoch': 1.5}
{'loss': 1.2997, 'grad_norm': 34.85115432739258, 'learning_rate': 2.464181414529809e-05, 'epoch': 1.75}
{'loss': 1.3334, 'grad_norm': 38.83604049682617, 'learning_rate': 2.25e-05, 'epoch': 2.0}
{'loss': 0.0847, 'grad_norm': 45.74128723144531, 'learning_rate': 2.0130302149885033e-05, 'epoch': 2.25}
{'loss': 0.3715, 'grad_norm': 16.787670135498047, 'learning_rate': 1.760472266500396e-05, 'epoch': 2.5}
{'loss': 0.5135, 'grad_norm': 19.38922691345215, 'learning_rate': 1.5e-05, 'epoch': 2.75}
{'loss': 0.35, 'grad_norm': 41.94072341918945, 'learning_rate': 1.2395277334996045e-05, 'epoch': 3.0}
{'loss': 0.0664, 'grad_norm': 9.1012544631958, 'learning_rate': 9.86969785011497e-06, 'epoch': 3.25}
{'loss': 0.1565, 'grad_norm': 18.93917465209961, 'learning_rate': 7.500000000000004e-06, 'epoch': 3.5}
{'loss': 0.0009, 'grad_norm': 0.24736496806144714, 'learning_rate': 5.358185854701911e-06, 'epoch': 3.75}
{'loss': 0.0916, 'grad_norm': 18.44576644897461, 'learning_rate': 3.5093333532153316e-06, 'epoch': 4.0}
{'loss': 0.1039, 'grad_norm': 22.31833267211914, 'learning_rate': 2.0096189432334194e-06, 'epoch': 4.25}
{'loss': 0.0339, 'grad_norm': 7.895977973937988, 'learning_rate': 9.046106882113753e-07, 'epoch': 4.5}
{'loss': 0.0564, 'grad_norm': 5.960275650024414, 'learning_rate': 2.278837048168797e-07, 'epoch': 4.75}
{'loss': 0.0003, 'grad_norm': 0.0744776725769043, 'learning_rate': 0.0, 'epoch': 5.0}
{'train_runtime': 379.4767, 'train_samples_per_second': 0.053, 'train_steps_per_second': 0.053, 'train_loss': 0.8549732926898287, 'epoch': 5.0}
***** train metrics *****
  epoch                    =        5.0
  total_flos               =     6139GF
  train_loss               =      0.855
  train_runtime            = 0:06:19.47
  train_samples_per_second =      0.053
  train_steps_per_second   =      0.053
```

### 5. 推理

这里问了一个问题,`"你帮我回答一下 Capybara 翻译软件的主要特性有什么?"`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cpu" # the device to load the model onto

def inference_sft_model(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

model_path = "models/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# run the inference and compare the result with the model before our finetuning
prompt = "你帮我回答一下 Capybara 翻译软件的主要特性有什么?"
print('Input:', prompt)
response = inference_sft_model(model, tokenizer, prompt)
print("Before fine-tuning:", response)

# load the fine-tuned model
model_path = "saves/Qwen2-0.5B-sft2"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)
response = inference_sft_model(model, tokenizer, prompt)
print("After fine-tuning:", response)
```

#### 5.1 微调之前的结果

```
Input: 你帮我回答一下 Capybara 翻译软件的主要特性有什么?
Before fine-tuning: Capybara 是一个用于 Ruby 的轻量级网络代理工具，它允许您在不需要安装代理的情况下进行网络连接。以下是 Capybara 主要特性的描述：

1. **可移植性**：Capybara 支持多种编程语言，包括但不限于 Ruby、Python 和 PHP，因此可以适用于不同的开发环境。

2. **跨平台支持**：Capybara 适配多种操作系统和浏览器，如 MacOS、Windows、Linux 和 Chrome 浏览器等。

3. **高效且简洁**：Capybara 可以提供高性能的网络代理服务，并且具有易于使用的设计原则。

4. **安全性和隐私**：Capybara 提供了高度安全的网络连接，包括加密通信和身份验证，保护用户的敏感信息。

5. **跨源请求处理**：Capybara 支持对不同来源的请求进行代理，从而避免不必要的网络流量消耗。

6. **灵活的配置管理**：Capybara 提供了简单的配置管理方式，用户可以根据需要添加或删除代理服务器，简化了配置过程。

7. **API接口支持**：Capybara 支持多种 API 接口，例如 HTTP 协议、HTTPS 协
```

#### 5.2 微调之后的结果

微调之前的模型，已经能做到根据我们的指令回答我们想要的结果了。但是回答的内容并不是想要的，微调之后，就能满足要求了。

```
After fine-tuning: Capybara 翻译软件的主要特性:

1. 多平台翻译比对：支持多种翻译平台间的翻译结果对比，帮助用户选择最佳翻译。
2. 强大的 OCR 识别：准确提取桌面展示图片的文字内容。
3. 跨平台支持：兼容 windows 和 macOS 系统，提供更好的使用体验。
4. 性能极佳：采用 Rust 开发的桌面应用，性能优越。
5. AI 驱动：利用人工智能技术提升翻译质量和用户体验。
```
