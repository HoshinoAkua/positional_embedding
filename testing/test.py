import transformers
from gpt2_modified import *
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, GPT2Config, Trainer
import numpy as np
import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'
import copy


model_path = '/mntcephfs/data/ruoyusun/common_dirs/gpt2-small'
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_from_disk('/mntcephfs/data/ruoyusun/hanyizhou/data/c4_512')
from tqdm import tqdm
import torch
ds = {}
for length in [512, 1024, 2048, 3072, 4096, 6144, 8192]:
    ds[length] = []
for i in tqdm(range(0,2500,50)): 
    data = sum(dataset[i:50+i]['input_ids'],[])
    for length in [512, 1024, 2048, 3072, 4096, 6144, 8192]:
        input_ids = data[:length]
        ds[length].append(input_ids)
#模型config设置
config = GPT2Config()

config.resid_pdrop = 0.0
config.embd_pdrop = 0.0
config.attn_pdrop = 0.0
config.vocab_size = 50258

model_fire = {}
model_cope = {}
model_mix = {}
#model_change = {}
for length in [512, 1024, 2048, 3072, 4096, 6144, 8192]:
    config.n_positions = length
    fire = GPT2_FIRE.from_pretrained('./output/fire/checkpoint-41250', config = config, ignore_mismatched_sizes=True)
    fire.transformer.wpe = torch.nn.Embedding(length,768)
    fire.transformer.wpe.weight = torch.nn.Parameter(torch.zeros(length,768))
    cope = GPT2_CoPE.from_pretrained('./output/cope/checkpoint-41250', config = config, ignore_mismatched_sizes=True,n_posmax = 384)
    mix = GPT2_FireWithCoPE.from_pretrained('./output/fire_with_cope/checkpoint-41250', config = config, ignore_mismatched_sizes=True)
    cope.transformer.wpe = torch.nn.Embedding(length,768)
    cope.transformer.wpe.weight = torch.nn.Parameter(torch.zeros(length,768))
    mix.transformer.wpe = torch.nn.Embedding(length,768)
    mix.transformer.wpe.weight = torch.nn.Parameter(torch.zeros(length,768))
    #change = copy.deepcopy(mix)
    # keys = fire.transformer._modules.keys()
    # for key in keys:
    #     if key != 'h':
    #         change.transformer._modules[key] = fire.transformer._modules[key]
    #     else:
    #         for i in range(12):
    #             change.transformer.h[i].attn.c_attn = fire.transformer.h[i].attn.c_attn
    #             change.transformer.h[i].attn.c_proj = fire.transformer.h[i].attn.c_proj
                
    # change.lm_head = fire.lm_head
    model_fire[length] = fire.to(device)
    model_cope[length] = cope.to(device)
    model_mix[length] = mix.to(device)
    #model_change[length] = change.to(device)
    
acc_fire, acc_cope, acc_mix, acc_change = {}, {}, {}, {}
for length in [512, 1024, 2048, 3072, 4096, 6144, 8192]:
    acc_fire[length] = []
    acc_cope[length] = []
    acc_mix[length] = []
    #acc_change[length] = []
for length in tqdm([512, 1024, 2048, 3072, 4096, 6144, 8192]):
    sequence_list = torch.tensor(ds[length]).to(device)
    fire = model_fire[length]
    cope = model_cope[length]
    mix = model_mix[length]
    #change = model_change[length]
    for sequence in sequence_list:
        with torch.no_grad():
            fire_output = fire(input_ids = sequence, labels = sequence)
        with torch.no_grad():
            cope_output = cope(input_ids = sequence, labels = sequence)
        with torch.no_grad():
            mix_output = mix(input_ids = sequence, labels = sequence)
        # with torch.no_grad():
        #     change_output = change(input_ids = sequence, labels = sequence)
        acc_fire[length].append(fire_output.loss.cpu())
        acc_cope[length].append(cope_output.loss.cpu())
        acc_mix[length].append(mix_output.loss.cpu())
        #acc_change[length].append(change_output.loss.cpu())


np.save('./output/acc_cope.npy', acc_cope) # type: ignore
np.save('./output/acc_fire.npy', acc_fire)
np.save('./output/acc_mix.npy', acc_mix)
#np.save('./output/acc_change.npy', acc_change)