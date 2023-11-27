import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from torch.nn import CrossEntropyLoss


# Função que realiza o cálculo da perplexidade
def compute_ppl(predictions, model, tokenizer, batch_size: int = 8, add_start_token: bool = True,
                max_length=None
                ):
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())

        assert (
                len(existing_special_tokens) > 0
        ), ("If batch_size > 1, model must have at least one special token to use for padding. Please use a different "
            "model or set batch_size=1.")

        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:

        assert (
                tokenizer.bos_token is not None
        ), ("Input model must already have a BOS token if using add_start_token=True. Please use a different model, "
            "or set add_start_token=False")
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(torch.cuda.current_device())

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), ("When add_start_token=False, each input text must be at least two tokens long. Run with "
            "add_start_token=True if inputting strings of only one token, and remove all empty input strings.")

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(
                torch.cuda.current_device())
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(torch.cuda.current_device()), attn_mask],
                dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            try:
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits
            except:
                out_logits = model(encoded_batch[:,:1024], attention_mask= attn_mask[:,:1024]).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


# Possíveis argumentos para chamar via linha de comando

parser = argparse.ArgumentParser(description="LLM Predictions")
parser.add_argument(
    "--dataset-name",
    required=True,
    type=str,
    help="Caminho do dataset no HuggingFace",
)

parser.add_argument(
    "--model-name",
    required=True,
    type=str,
    help="O modelo para calcular as métricas",
)

parser.add_argument(
    "--output-dir",
    required=False,
    type=str,
    default='./results',
    help="Caminho para salvar os resultados",
)

parser.add_argument(
    "--checkpoint-path",
    required=True,
    type=str,
    default='',
    help="Caminho para salvar os resultados",
)

parser.add_argument(
    "--output-file",
    required=True,
    type=str,
    default='',
    help="Nome do arquivo de csv de saída",
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

    dataset_teste = load_dataset(args.dataset_name)['train']  # carregando o dataset de teste

    # Carregando o modelo e tokenizador:

    model_name = ""

    if args.model_name == "zephyr":
        model_name = "HuggingFaceH4/zephyr-7b-beta"


    elif args.model_name == "gpt2":
        model_name = "gpt2"

    elif args.model_name == "distilgpt2":
        model_name = "distilgpt2"

    # Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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
        model_name, quantization_config=bnb_config, device_map={"": torch.cuda.current_device()}
    )

    # Configurando o token de padding no modelo
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.eval()

    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, args.checkpoint_path)

    results = {}  # dicionário que irá armazenar os resultados no formato 'nome do dataset': 'quantidade de acerto'

    # os datasets que são questões de múltipla escolha
    multiples = ['MedMCQA', 'MedQA (USMLE)', 'MMLU/anatomy', 'MMLU/clinical_knowledge',
                 'MMLU/college_medicine', 'MMLU/medical_genetics', 'MMLU/professional_medicine',
                 'MMLU/college_biology']

    # Inicializando a quantidade de acerto dos datasets
    for m in multiples:
        results[m] = 0

    results['PubMedQA'] = 0  # Único dataset que não segue o mesmo formato de múltipla escolha

    letras = ['A', 'B', 'C', 'D']  # lista para extrair a letra a partir da resposta

    # Iterando sobre o dataset de teste
    for i in tqdm(range(len(dataset_teste))):

        # Verificando se é um dataset de múltipla escolha com letras A, B, C e D.
        if dataset_teste[i]['dataset'] in multiples:

            # Armazenando as possíveis alternativas
            alternativas = [dataset_teste[i]['alternative_a'], dataset_teste[i]['alternative_b'],
                            dataset_teste[i]['alternative_c'], dataset_teste[i]['alternative_d']]

            ppls = []  # lista para armazenar a perplexidade de cada alternativa, visto que a menor perplexidade
            # corresponde à correta

            # Iterando sobre as alternativas
            for alt in alternativas:

                # Montando o prompt:
                system_prompt = ('Abaixo está uma instrução que descreve uma tarefa, junto com o input que oferece um '
                                 'contexto adicional. Retorne uma resposta apropriada ao pedido.\n\n')
                B_INST, E_INST = "### Instrução:\n", "### Resposta:\n"

                x = dataset_teste[i]['instruction']
                x += f"A) {dataset_teste[i]['alternative_a']}"
                x += f"B) {dataset_teste[i]['alternative_b']}"
                x += f"C) {dataset_teste[i]['alternative_c']}"
                x += f"D) {dataset_teste[i]['alternative_d']}"

                prompt = f"{system_prompt}{B_INST}{x.strip()}\n\n{E_INST}"
                prompt += alt

                # Calculando a perplexidade:
                perplexities = compute_ppl(predictions=prompt, model=model, tokenizer=tokenizer,
                                           add_start_token=False, max_length=1024)

                ppls.append(perplexities['perplexities'][0])  # Armazenando a perplexidade

            else:

                # Verificando se o modelo acertou ou não a resposta
                pred = letras[np.argmin(ppls)]
                true = dataset_teste[i]['output']
                if pred == true:
                    results[dataset_teste[i]['dataset']] += 1

        # Quando o dataset é o PubMedQA:
        else:
            alternativas = ['sim', 'talvez', 'não'] # possíveis respostas

            ppls = [] # armazenando as perplexidades

            for alt in alternativas:

                # Montando o prompt:
                system_prompt = ('Abaixo está uma instrução que descreve uma tarefa, junto com o input que oferece um '
                                 'contexto adicional. Retorne uma resposta apropriada ao pedido.\n\n')
                B_INST, E_INST = "### Instrução:\n", "### Resposta:\n"

                x = dataset_teste[i]['instruction']

                prompt = f"{system_prompt}{B_INST}{x.strip()}\n\n###Input:{dataset_teste[i]['input']}\n\n{E_INST}"
                prompt += alt

                # Calculando a perplexidade:
                perplexities = compute_ppl(predictions=prompt, model=model, tokenizer=tokenizer,
                                           add_start_token=False, max_length=1024)

                ppls.append(perplexities['perplexities'][0])  # Armazenando a perplexidade

            else:

                # Verificando se o modelo acertou ou não a resposta
                pred = alternativas[np.argmin(ppls)]
                true = dataset_teste[i]['output']
                if pred == true:
                    results[dataset_teste[i]['dataset']] += 1

    # Extraindo o tamanho de cada dataset:
    df = dataset_teste.to_pandas()
    features = ["dataset"]
    df = df[features]
    tamanho_dataset = dict(df.dataset.value_counts())

    # Normalizando a quantidade de acerto pelo tamanho de cada dataset
    for key, value in results.items():
        results[key] = value / tamanho_dataset[key]

    # Salvando os resultados
    df = pd.DataFrame(list(results.items()), columns=['dataset', 'porcentagem_acerto'])

    df.to_csv(f'{args.output_dir}/{args.output_file}.csv', index=False)
    print("Predições salvas em: ", f'{args.output_dir}/{args.output_file}.csv !')


if __name__ == "__main__":
    run()
