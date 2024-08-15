from accelerate import Accelerator
from datasets import load_from_disk
import wandb
from transformers import AutoTokenizer, TrainingArguments, GPT2Config, Trainer
from transformers import DataCollatorForLanguageModeling
import os


wandb.login()
os.environ["WANDB_PROJECT"]="fire"
model_path = './output/fire/checkpoint-30000'
tokenizer = AutoTokenizer.from_pretrained( '/mntcephfs/data/ruoyusun/common_dirs/gpt2-small')
dataset = load_from_disk('/mntcephfs/data/ruoyusun/hanyizhou/data/c4_512')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
from gpt2_modified import GPT2_FIRE
accelerator = Accelerator()
config = GPT2Config()
config.n_positions = 512
config.resid_pdrop = 0.0
config.embd_pdrop = 0.0
config.attn_pdrop = 0.0
config.vocab_size = 50258
model_fire = GPT2_FIRE(config)
#model_row = GPT2LMHeadModel(config)

args_fire = TrainingArguments(
    output_dir='./output/fire',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate= 6e-4,
    num_train_epochs=3,
    report_to = 'wandb', # type: ignore
    logging_steps = 1,
    save_steps = 5000
)
trainer_fire = Trainer(
    args = args_fire,
    model = model_fire,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset = dataset# type: ignore
)
trainer = accelerator.prepare(trainer_fire)
trainer.train(resume_from_checkpoint=model_path)
