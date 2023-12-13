from llm.GepetoBase import ChatGPT


class LongAnswerAnalysis:
    def __init__(self, key_path):
        self.gpt = ChatGPT(key_path)

    def first_criterion(self, pergunta, resposta, resposta_esperada):
        prompt = f"Dada a seguinte pergunta relacionada à área da saúde, marcada por <pergunta>, a resposta fornecida por uma Inteligência Artificial, marcada por <resposta>, e a resposta real da pergunta, marcada por <real>, responda a dúvida a seguir, marcada por <duvida>.\n\n <pergunta> {pergunta} <pergunta> \n <resposta> {resposta} <resposta> \n <real> {resposta_esperada} <real> \n\n <duvida> Dúvida: Quão bem a resposta, dada pela Inteligência Artificial, aborda a intenção pergunta? Selecione uma das alternativas a seguir.\n <duvida> 1- Aborda a intenção da pergunta diretamente. (5 pontos)\n 2- Aborda a intenção da pergunta indiretamente. (3 pontos)\n 3- Não aborda a intenção da pergunta. (0 pontos)\n\n A partir da alternativa selecionada, retorne apenas sua pontuação numérica, que está entre parênteses após a alternativa, e nada mais.\n Exemplo de saída (caso a primeira alternativa tenha sido selecionada): 5\n Exemplo de saída (caso a segunda alternativa tenha sido selecionada): 3\n Exemplo de saída (caso a terceira alternativa tenha sido selecionada): 0\n"

        self.gpt.add_message(role="user", prompt=prompt)

        # fazendo inferencia
        resposta, etc = self.gpt.get_completion()

        resposta = eval(resposta)

        self.gpt.reset_messages(maintain_context=False)

        return resposta

    def second_criterion(self, pergunta, resposta, resposta_esperada):
        prompt = f"Dada a seguinte pergunta relacionada à área da saúde, marcada por <pergunta>, a resposta fornecida por uma Inteligência Artificial, marcada por <resposta>, e a resposta real da pergunta, marcada por <real>, responda a dúvida a seguir, marcada por <duvida>.\n\n <pergunta> {pergunta} <pergunta> \n <resposta> {resposta} <resposta> \n <real> {resposta_esperada} <real> \n\n <duvida> Dúvida: Quão útil a resposta é para o usuário? Ou seja, ela permite chegar a uma conclusão ou ter uma noção dos próximos passos? Selecione uma das alternativas a seguir. <duvida>\n 1- Extremamente útil. (5 pontos)\n 2- Útil. (3 pontos)\n 3- Um pouco útil. (1 ponto)\n 4- Não é útil. (0 pontos)\n\n A partir da alternativa selecionada, retorne apenas sua pontuação numérica, que está entre parênteses após a alternativa, e nada mais.\n  Exemplo de saída (caso a primeira alternativa tenha sido selecionada): 5\n Exemplo de saída (caso a segunda alternativa tenha sido selecionada): 3\n"

        self.gpt.add_message(role="user", prompt=prompt)

        # fazendo inferencia
        resposta, etc = self.gpt.get_completion()
        resposta = eval(resposta)

        self.gpt.reset_messages(maintain_context=False)

        return resposta

    def third_criterion(self, pergunta, resposta, resposta_esperada):
        prompt = f"Dada a seguinte pergunta relacionada à área da saúde, marcada por <pergunta>, a resposta fornecida por uma Inteligência Artificial, marcada por <resposta>, e a resposta real da pergunta, marcada por <real>, responda a dúvida a seguir, marcada por <duvida>.\n\n <pergunta> {pergunta} <pergunta> \n <resposta> {resposta} <resposta> \n <real> {resposta_esperada} <real> \n\n <duvida> Dúvida: A resposta apresenta uma boa coesão e coerência, ou seja, as partes do texto estão bem vinculadas e fazem sentido? Selecione uma das alternativas a seguir. <duvida>\n 1- O texto é coeso e coerente. (5 pontos)\n 2- O texto é parcialmente coeso e coerente. (3 pontos)\n 3- O texto não apresenta coesão e não é coerente. (0 pontos)\n\n A partir da alternativa selecionada, retorne apenas sua pontuação numérica, que está entre parênteses após a alternativa, e nada mais.\n Exemplo de saída (caso a primeira alternativa tenha sido selecionada): 5\n Exemplo de saída (caso a segunda alternativa tenha sido selecionada): 3\n Exemplo de saída (caso a terceira alternativa tenha sido selecionada): 0\n"

        self.gpt.add_message(role="user", prompt=prompt)

        # fazendo inferencia
        resposta, etc = self.gpt.get_completion()
        resposta = eval(resposta)

        self.gpt.reset_messages(maintain_context=False)

        return resposta

    def all_criterions(self, pergunta, resposta, resposta_esperada):
        r1 = self.first_criterion(pergunta, resposta, resposta_esperada)
        r2 = self.second_criterion(pergunta, resposta, resposta_esperada)
        r3 = self.third_criterion(pergunta, resposta, resposta_esperada)

        return [r1, r2, r3]