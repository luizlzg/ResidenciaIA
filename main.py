import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
import argparse
import os

parser = argparse.ArgumentParser(description="LLM Finetuning")
parser.add_argument(
    "--dataset-name",
    required=True,
    type=str,
    help="Caminho do dataset no HuggingFace",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="cuda",
    type=str,
    help='Device ID for your GPU (just pass the device ID number). (default: "cuda")',
)
parser.add_argument(
    "--model-name",
    required=True,
    default="zephyr",
    type=str,
    choices=["zephyr", "distilgpt2", "gpt2"],
    help="O modelo para se fazer o finetuning",
)

parser.add_argument(
    "--lora-r",
    required=False,
    type=int,
    default=16,
    help='Valor do parâmetro r.',
)
parser.add_argument(
    "--lora-alpha",
    required=False,
    type=int,
    default=16,
    help='Valor do parâmetro lora_alpha.',
)
parser.add_argument(
    "--lora-dropout",
    required=False,
    type=float,
    default=0.1,
    help='Valor do parâmetro lora_dropout.',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=16,
    help="O tamanho do batch para treino e validação. (default: 16)",
)
parser.add_argument(
    "--num-train-epochs",
    required=False,
    type=int,
    default=2,
    help="A quantidade de épocas de treino",
)
parser.add_argument(
    "--output-dir",
    required=False,
    type=str,
    default='./results',
    help="Caminho para salvar os resultados",
)

parser.add_argument(
    "--log-dir",
    required=False,
    type=str,
    default='./logs',
    help="Caminho para salvar os logs",
)


def run():
    args = parser.parse_args()

    output_path = args.output_dir
    if not os.path.exists(output_path):
        print(f"O diretório {output_path} não existe. Criando...")
        try:
            # criando o diretório de output
            os.makedirs(output_path)
        except Exception as e:
            print(f"Não foi possível criar o diretório {output_path}. Erro: {e}")
            exit(1)

    if not os.path.exists(args.log_dir):
        print(f"O diretório {args.log_dir} não existe. Criando...")
        try:
            # criando o diretório de output
            os.makedirs(args.log_dir)
        except Exception as e:
            print(f"Não foi possível criar o diretório {args.log_dir}. Erro: {e}")
            exit(1)

    dataset = load_dataset(args.dataset_name)

    if args.model_name == "zephyr":
        modelo = "HuggingFaceH4/zephyr-7b-beta"
        tokenizer = AutoTokenizer.from_pretrained(modelo, use_fast=True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = 'right'

        # Quantização
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        # Carregando o modelo e quantizando
        model = AutoModelForCausalLM.from_pretrained(
            modelo, quantization_config=bnb_config, device_map={"": torch.cuda.current_device()}
        )

        # Transforma alguns módulos do modelo para o tipo de dado fp32
        model = prepare_model_for_kbit_training(model)

        # Configurando o token de padding no modelo
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False  # Gradient checkpointing is used by default but not compatible with caching
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
        )

        training_arguments = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=args.num_train_epochs,
            evaluation_strategy="steps",
            do_eval=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            optim="paged_adamw_8bit",
            save_steps=200,
            logging_steps=100,
            learning_rate=2e-4,
            logging_dir=args.log_dir,
            max_steps=7000,
            warmup_steps=700,
            lr_scheduler_type="linear",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        trainer.train()

    elif args.model_name == "distilgpt2":
        modelo = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(modelo, use_fast=True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = 'right'

        # Quantização
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        # Carregando o modelo e quantizando
        model = AutoModelForCausalLM.from_pretrained(
            modelo, quantization_config=bnb_config, device_map={"": torch.cuda.current_device()}
        )

        # Transforma alguns módulos do modelo para o tipo de dado fp32
        model = prepare_model_for_kbit_training(model)

        # Configurando o token de padding no modelo
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False  # Gradient checkpointing is used by default but not compatible with caching
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
        )

        training_arguments = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=args.num_train_epochs,
            evaluation_strategy="steps",
            do_eval=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            optim="paged_adamw_8bit",
            save_steps=200,
            logging_steps=100,
            learning_rate=2e-4,
            logging_dir=args.log_dir,
            max_steps=7000,
            warmup_steps=700,
            lr_scheduler_type="linear",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        trainer.train()

    elif args.model_name == "gpt2":
        modelo = "gpt2"

        tokenizer = AutoTokenizer.from_pretrained(modelo, use_fast=True)

        tokenizer.pad_token = tokenizer.unk_token

        tokenizer.pad_token_id = tokenizer.unk_token_id

        tokenizer.padding_side = 'right'

        # Quantização

        compute_dtype = getattr(torch, "float16")

        bnb_config = BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=compute_dtype,

            bnb_4bit_use_double_quant=True,

        )

        # Carregando o modelo e quantizando

        model = AutoModelForCausalLM.from_pretrained(

            modelo, quantization_config=bnb_config, device_map={"": torch.cuda.current_device()}

        )

        # Transforma alguns módulos do modelo para o tipo de dado fp32

        model = prepare_model_for_kbit_training(model)

        # Configurando o token de padding no modelo

        model.config.pad_token_id = tokenizer.pad_token_id

        model.config.use_cache = False  # Gradient checkpointing is used by default but not compatible with caching

        model.config.pretraining_tp = 1

        peft_config = LoraConfig(

            lora_alpha=args.lora_alpha,

            lora_dropout=args.lora_dropout,

            r=args.lora_r,

            bias="none",

            task_type="CAUSAL_LM",

            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]

        )

        training_arguments = TrainingArguments(

            output_dir=output_path,

            num_train_epochs=args.num_train_epochs,

            evaluation_strategy="steps",

            do_eval=True,

            per_device_train_batch_size=args.batch_size,

            per_device_eval_batch_size=args.batch_size,

            gradient_accumulation_steps=2,

            optim="paged_adamw_8bit",

            save_steps=200,

            logging_steps=100,

            learning_rate=2e-4,

            logging_dir=args.log_dir,

            max_steps=7000,

            warmup_steps=700,

            lr_scheduler_type="linear",

        )

        trainer = SFTTrainer(

            model=model,

            train_dataset=dataset['train'],

            eval_dataset=dataset['validation'],

            peft_config=peft_config,

            dataset_text_field="text",

            max_seq_length=512,

            tokenizer=tokenizer,

            args=training_arguments,

        )

        trainer.train()


if __name__ == "__main__":
    run()
