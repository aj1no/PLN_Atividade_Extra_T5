# Tradução Automática Inglês-Português com PTT5 e OPUS-100 - PLN (Atividade Extra)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/PLN_Atividade_Extra_T5/blob/main/NotebookExtra_T5.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.9.5-792EE5?style=flat-square&logo=lightning&logoColor=white)](https://lightning.ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![SacreBLEU](https://img.shields.io/badge/Metric-SacreBLEU-blue?style=flat-square)](https://github.com/mjpost/sacrebleu)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Este repositório contém a resolução da **Atividade Extra: Fine-Tuning do Modelo T5 (PTT5) para Tradução Inglês-Português**, desenvolvida para a disciplina de **Processamento de Linguagem Natural (PLN)**.

---

## Informações Acadêmicas

* **Curso:** Ciência de Dados (5º Semestre)
* **Disciplina:** Processamento de Linguagem Natural (PLN)
* **Professor:** Leonardo de Lellis Rossi
* **Aluno:** Rodolfo Vinicius Cima Takemoto
* **Atividade:** Atividade Extra (Treinamento do Modelo T5 / PTT5 em Tradução Automática EN -> PT)

---

## Objetivo e Visão Geral

O objetivo principal desta atividade é realizar o fine-tuning supervisionado da arquitetura **T5 (Text-to-Text Transfer Transformer)** utilizando a variante **PTT5** ([unicamp-dl/ptt5-small-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-small-portuguese-vocab)), um modelo pré-treinado em corpus da língua portuguesa com vocabulário dedicado.

### Principais Características da Abordagem:
1. **Dataset OPUS-100 (`en-pt`):**
   - Treinamento: 100.000 pares de sentenças inglês $\rightarrow$ português.
   - Validação: 5.000 pares de sentenças.
   - Teste: 5.000 pares de sentenças.
2. **Tokenização e Codificação:**
   - `T5Tokenizer` com truncamento e preenchimento controlado (`max_length=128`).
   - Mascaramento de rótulos com índice `-100` nas posições de padding para desconsiderar a perda sobre tokens nulos.
3. **Métrica SacreBLEU:**
   - Avaliação padronizada de tradução automática via `sacrebleu.corpus_bleu`, garantindo reprodutibilidade e pré-processamento uniforme de texto.
4. **PyTorch Lightning (`pl.LightningModule`):**
   - Estruturação modular contendo passos de treinamento, validação com geração de hipóteses (`model.generate`), registro automático de métricas (`avg_val_loss`, `avg_val_bleu`), checkpointing e acumulação de gradientes (`accumulate_grad_batches=2`).

---

## Estrutura do Notebook

O notebook [`NotebookExtra_T5.ipynb`](NotebookExtra_T5.ipynb) está organizado nas seguintes etapas:

### 1. Configurações Gerais e Dependências
* Definição de hiperparâmetros globais: `model_name="unicamp-dl/ptt5-small-portuguese-vocab"`, `batch_size=64`, `accumulate_grad_batches=2`, `learning_rate=1e-3`, `max_length=128`.
* Instalação de pacotes: `sacrebleu`, `pytorch-lightning==1.9.5`, `transformers`, `sentencepiece`, `datasets`, `nvidia-ml-py`.

### 2. Reprodutibilidade e Hardware
* Fixação de sementes (`seed=123`) em `random`, `torch` e `torch.cuda`.
* Monitoramento de consumo de VRAM da GPU via `pynvml`.
* Montagem do Google Drive para persistência de checkpoints (`epoch=1.ckpt`).

### 3. Extração e Divisão do Dataset OPUS-100
* Carregamento streaming do dataset `Helsinki-NLP/opus-100` (`en-pt`).
* Amostragem e embaralhamento reprodutível (`seed=42`) dividindo o conjunto em treino (100k), validação (5k) e teste (5k).

### 4. Implementação da Classe `MyDataset`
* Subclasse de `torch.utils.data.Dataset` customizada.
* Codificação paralela de sentenças de origem (inglês) e destino (português) gerando `source_token_ids`, `source_mask`, `target_token_ids`, `target_mask` e textos originais.

### 5. Arquitetura `T5Finetuner` (PyTorch Lightning)
* **`forward`:** Suporte dual para treinamento supervisionado com cálculo de perda ou inferência via decodificação autoregressiva (`generate`).
* **`training_step`:** Execução do passo de forward e logging em tempo real de `train_loss`.
* **`validation_step` & `test_step`:** Cálculo simultâneo da perda e geração de texto para decodificação em lote (`tokenizer.batch_decode`).
* **`validation_epoch_end` & `test_epoch_end`:** Cálculo agregado da pontuação **SacreBLEU** sobre o corpus completo de validação/teste e exibição de amostras comparativas (Alvo vs Predição).
* **`configure_optimizers`:** Otimizador Adam com taxa de aprendizado constante e monitoramento da métrica `avg_val_bleu`.

### 6. Validação e Overfitting em Mini-Batch
* **Fast Dev Run (`fast_dev_run=True`):** Verificação de integridade de todo o pipeline em 1 lote de treino/validação/teste.
* **Overfitting Controlado (`overfit_batches=0.005`):** Teste de sanidade do modelo em lote restrito para validar convergência da perda a 0.

### 7. Treinamento Completo e Avaliação Final
* Treinamento por múltiplas épocas com salvamento de checkpoints automáticos.
* Avaliação final cega no conjunto de teste (`trainer.test`) com métrica final de SacreBLEU e relatório de traduções.

---

## Ferramentas Utilizadas

Em conformidade com as diretrizes de transparência acadêmica:

* **Ferramenta:** Google Colab / Antigravity IDE
  * **Utilização:** Ambiente de desenvolvimento, execução dos scripts e edição de notebooks.
* **Ferramenta:** Gemini 3.7 Flash (High) / Antigravity
  * **Utilização:** Desenvolvimento e implementação dos métodos de `MyDataset` e `T5Finetuner` (PyTorch Lightning), integração da métrica SacreBLEU e estruturação da documentação.
* **Ferramenta:** Git / GitHub
  * **Utilização:** Controle de versão e publicação do repositório acadêmico.

---

## Como Executar

### Opção 1: Execução no Google Colab (Recomendado)
Clique no badge abaixo para abrir e executar diretamente com GPU T4/A100 no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/PLN_Atividade_Extra_T5/blob/main/NotebookExtra_T5.ipynb)

### Opção 2: Execução Local

1. Clone o repositório:
   ```bash
   git clone https://github.com/aj1no/PLN_Atividade_Extra_T5.git
   cd PLN_Atividade_Extra_T5
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   .\venv\Scripts\activate   # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install torch torchvision transformers sentencepiece sacrebleu pytorch-lightning==1.9.5 datasets nvidia-ml-py
   ```

4. Inicie o Jupyter Notebook:
   ```bash
   jupyter notebook NotebookExtra_T5.ipynb
   ```
