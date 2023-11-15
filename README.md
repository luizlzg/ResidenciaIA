# Descrição geral

Repositório destinado aos códigos desenvolvidos na disciplina de Residência em IA. Tais códigos estão relacionados ao pré-processamento de dados, criação de datasets, pós-processamento dos datasets todos relacionados ao finetuning de LLMs. Além disso, há códigos de suporte para o cálculo de benchmarks e, principalmente, os códigos para a realização do finetuning dos LLMs em si.

# Pré-processamento

Na pasta 'preprocess', encontram-se os notebooks utilizados para o pré-processamento dos dados, ou seja, o código para a realização da tradução dos dados captados. <br>
Além disso, na pasta 'llm', encontra-se o código de apoio para utilizar a API do ChatGPT, que foi utilizado para realizar a tradução.

# Dados

Na pasta 'dados' estão os dados de treino e teste, separados em suas respectivas pastas. Lembrando que os dados de testes são os dados utilizados para calcular as métricas do benchmark MultiMedQA.

# Métricas

Na pasta 'evaluate', encontra-se o código utilizado para calcular as métricas do benchmark. Mais especificamente, tal benchmark calcula a taxa de acerto em questões de múltiplas escolhas na área da saúde. <br>
Desse modo, para cada questão, e suas respectivas alternativas, foi calculada a perplexidade do modelo, sendo que a menor perplexidade significa o palpite do modelo. Sendo assim, a partir de tal palpite a taxa de acerto será calculada para cada dataset.
