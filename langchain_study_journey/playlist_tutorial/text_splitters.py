from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path

text = """A inteligência artificial (IA) é uma área da ciência da computação que tem revolucionado diversas \
indústrias e aspectos da vida cotidiana. Mas, o que realmente significa "inteligência artificial"? Trata-se de sistemas \
computacionais capazes de realizar tarefas que, anteriormente, só poderiam ser executadas por seres humanos, como \
reconhecimento de fala, tomada de decisão e aprendizado com dados. Impressionante, não é? Esses sistemas utilizam \
algoritmos avançados e grandes volumes de dados para identificar padrões, adaptarem-se a novas situações e fornecerem \
soluções inovadoras.

Um dos maiores avanços recentes em IA é o aprendizado profundo (ou deep learning). Essa técnica permite que máquinas \
realizem tarefas extremamente complexas, como diagnosticar doenças a partir de imagens médicas ou até mesmo compor \
músicas! Curioso como isso funciona? Redes neurais artificiais – inspiradas no funcionamento do cérebro humano – \
processam informações em múltiplas camadas, identificando nuances que seriam impossíveis para métodos tradicionais. \
Como resultado, a IA tem transformado áreas como saúde, finanças e transporte, promovendo eficiência e inovação em \
escala global.

No entanto, a expansão da inteligência artificial também levanta questões importantes. Estamos preparados para lidar \
com os desafios éticos que a IA traz? Por exemplo: como garantir que algoritmos de IA sejam imparciais e inclusivos? \
Além disso, há preocupações sobre o impacto no mercado de trabalho – algumas profissões podem ser substituídas por \
máquinas. Apesar desses desafios, uma coisa é certa: a inteligência artificial já não é mais uma tecnologia do futuro; \
é uma realidade do presente, moldando o mundo ao nosso redor com potencial ilimitado!"""

texto_original = Document(page_content=text)
docs = [texto_original]

# ## Exemplo 1: chunk gerado por comprimento de caracteres:

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,  # tamanho dos pedaços
    chunk_overlap=5,  # sobreposição de pedaços
    length_function=len,  # tipo de divisão: por caractere
    separators=[""],
)
texts = text_splitter.split_documents(docs)

for pedaco in texts:
    print(pedaco)


# ---------------------------------------------------------------------------------------------------------------------
## Exemplo 2: chunk gerado por comprimento de tokens:

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(separators=[""],
                                                                     encoding_name="cl100k_base",
                                                                     chunk_size=100,
                                                                     chunk_overlap=20)
texts = text_splitter.split_documents(docs)

for pedaco in texts:
    print(pedaco)
# ---------------------------------------------------------------------------------------------------------------------
## Exemplo 3: chunk gerado por caractere especifico (parágrafos):

text_splitter = CharacterTextSplitter(
    separator="\n\n",  # dividir por paragrafos
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(docs)

for pedaco in texts:
    print(pedaco)


# ---------------------------------------------------------------------------------------------------------------------
## Exemplo 4: chunk gerado por md:


caminho = Path(__file__).parent / "exemplo_markdown.md"
print(caminho)
with open(caminho) as f:
    arquivo = f.read()

# Mapeamento de quebras:
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Instanciando o separador:
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# Aplicando o separador:
resultado_com_split_de_cabecalho = markdown_splitter.split_text(arquivo)

for pedaco in resultado_com_split_de_cabecalho:
    print("--" * 30)
    print(pedaco)

# ---------------------------------------------------------------------------------------------------------------------
## Exemplo 5: chunk gerado por semantica


# Criando o splitter usando um modelo 'text-embedding-3-large' da OpenAI
text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Aplicando o split
texts = text_splitter.split_documents(docs)


for pedaco in texts:
    print("--" * 30)
    print(pedaco)