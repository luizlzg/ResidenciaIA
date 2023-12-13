from datasets import load_dataset
from tqdm import tqdm
from gpt_analyst import LongAnswerAnalysis
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import pandas as pd
import argparse
import os
import torch
from transformers import GenerationConfig


def generate_answer(instruction, tokenizer, model):
    runtimeFlag = torch.cuda.current_device()
    system_prompt = ('Abaixo está uma instrução/dúvida relacionada à saúde, vinda de um usuário. Retorne uma resposta '
                     'apropriada ao pedido.\n\n')
    B_INST, E_INST = "### Instrução:\n", "### Resposta:\n"

    prompt = f"{system_prompt}{B_INST}{instruction.strip()}\n\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    generation_output = model.generate(**inputs, max_new_tokens=512,
                                       generation_config=GenerationConfig(pad_token_id=tokenizer.pad_token_id),
                                       return_dict_in_generate=True,
                                       output_scores=True, )

    resposta = tokenizer.decode(generation_output.sequences[0]).split('### Resposta:')[1].split('\n\n')[0] + '\n\n'

    return resposta.strip()


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
    gpt4 = LongAnswerAnalysis('../key.txt')
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

    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    model.eval()

    results = []  # lista que irá armazenar os resultados no formato ['nome do modelo', 'id da pergunta', 'pergunta',
    # 'resposta', 'resposta esperada', 'primeiro critério', 'segundo critério', 'terceiro critério',
    # 'soma dos critérios']

    for i in tqdm(range(len(dataset_teste))):
        resposta = generate_answer(dataset_teste[i]['instruction'], tokenizer, model)
        criterios = gpt4.all_criterions(dataset_teste[i]['instruction'], resposta, dataset_teste[i]['output'])
        soma = criterios[0] + criterios[1] + criterios[2]

        results.append(
            [args.model_name, i, dataset_teste[i]['instruction'], resposta, dataset_teste[i]['output'], criterios[0],
             criterios[1], criterios[2], soma])

    # Salvando os resultados gerais
    df = pd.DataFrame(results,
                      columns=['model_name', 'id', 'pergunta', 'resposta', 'resposta_esperada', 'primeiro_criterio',
                               'segundo_criterio', 'terceiro_criterio', 'soma'])

    df.to_csv(f'{args.output_dir}/{args.output_file}.csv', index=False)
    print("Resultados salvos em: ", f'{args.output_dir}/{args.output_file}.csv !')

    # Salvando a média do resultado
    media = f"O modelo {args.model_name}, obteve um desempenho médio de {df.soma.mean()} pontos. Normalizado: {df.soma.mean() / 15} pontos."

    # Abre o arquivo em modo de escrita
    with open(f'{args.output_dir}/{args.output_file}_media.txt', 'w') as arquivo:
        # Escreve a string no arquivo
        arquivo.write(media)

    print(f"O resultado médio foi salvo em: {args.output_dir}/{args.output_file}_media.txt.")


if __name__ == "__main__":
    run()