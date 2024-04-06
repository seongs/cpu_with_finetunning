import time
import argparse

from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments)



def main(FLAGS):

    dataset = load_dataset("seongs/may2", split="train") #데이터셋

    model_name = "beomi/Yi-Ko-6B" #모델 이름
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

    training_arguments = TrainingArguments(
        output_dir="./results", #모델 저장 위치
        bf16=FLAGS.bf16, #change for CPU
        use_ipex=FLAGS.use_ipex, #change for CPU IPEX
        no_cuda=True,
        fp16_full_eval=False,
        per_device_train_batch_size=1,  
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        report_to="tensorboard"
    )

    

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=FLAGS.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True,
        peft_config=peft_params
    )


    start = time.time()

    trainer.train()

    total = time.time() - start

    print(f'Time to tune {total}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-bf16',
                        '--bf16',
                        type=bool,
                        default=False,
                        help="activate mix precision training with bf16")
    parser.add_argument('-ipex',
                        '--use_ipex',
                        type=bool,
                        default=True,
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-msq',
                        '--max_seq_length',
                        type=int,
                        default=512,
                        help="specifies the number of highest probability tokens to consider at each step")
    
    FLAGS = parser.parse_args()
    main(FLAGS)