from accelerate import Accelerator
from datasets import load_from_disk
import wandb
from transformers import AutoTokenizer, TrainingArguments, GPT2Config, Trainer
from transformers import DataCollatorForLanguageModeling
import os


wandb.login()
os.environ["WANDB_PROJECT"]="Fire with CoPE"
model_path = '/mntcephfs/data/ruoyusun/common_dirs/gpt2-small'
checkpoint_path = './output/fire_with_cope/checkpoint-37500'
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_from_disk('/mntcephfs/data/ruoyusun/hanyizhou/data/c4_512')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
from gpt2_modified import GPT2_FireWithCoPE
accelerator = Accelerator()
config = GPT2Config()
config.n_positions = 512
config.resid_pdrop = 0.0
config.embd_pdrop = 0.0
config.attn_pdrop = 0.0
config.vocab_size = 50258
model = GPT2_FireWithCoPE(config)
#model_cope = GPT2_RawCoPE(config, n_posmax=64)
#model_row = GPT2LMHeadModel(config)

args = TrainingArguments(
    output_dir='./output/fire_with_cope',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate= 1e-4,
    num_train_epochs=3,
    report_to = 'wandb', # type: ignore
    logging_steps = 2,
    save_steps = 1250
)
trainer = Trainer(
    args = args,
    model = model,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset = dataset# type: ignore
)

trainer = accelerator.prepare(trainer)
trainer.train(resume_from_checkpoint = checkpoint_path)
