import re
from datetime import datetime, timedelta
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from bertopic import BERTopic
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, util

palavras_interrogativas = {
    "quem", "o que", "qual", "quais", "quando",
    "onde", "como", "por que", "porque", "pra que",
    "será que", "alguém sabe", "tem como"
}
padroes_ruido = [
    r'Messages and calls are end-to-end encrypted.\..*',  # Mensagem de criptografia
    r'You (deleted|edited) this message.*',              # Mensagens apagadas/editadas
    r'This message was deleted.*',                       # Mensagem apagada por outro
    r'Missed (voice|video) call.*',                      # Chamadas perdidas
    r'created group.*',                                  # Criação de grupo
    r'\b\w+ (added|removed|left|changed).*',            # Ações no grupo
    r'<Media omitted>',                                  # Mídias não baixadas
    r'(image|audio|video|sticker|document) omitted',     # Mídias específicas
    r'‎']     


def is_question_f(mensagem):
    mensagem_lower = mensagem.lower().strip()
    
    if mensagem_lower.endswith("?"):
        return True
    
    for palavra in palavras_interrogativas:
        if mensagem_lower.startswith(palavra):
            return True

    return False


def parse_message(message):
    # Padrão regex para extrair data, hora, autor e conteúdo
    pattern = r'\[(\d{2}/\d{2}/\d{2}), (\d{2}:\d{2}:\d{2})\] (.+?): (.+)'
    match = re.search(pattern, message)
 

    if match:
        date = match.group(1)
        time = match.group(2)
        author = match.group(3).strip()
        content = match.group(4).strip()
        is_question =  is_question_f(content)
        # if any(padrao in content for padrao in padroes_ruido):

        #     return None
        return {
            'date': date,
            'time': time,
            'author': author,
            'content': content,
            'is_question' : is_question

        }
    return None


def extract_info(df : pd.DataFrame, embeddings_model : SentenceTransformer = None, max_response_hours : int =2, similarity_threshold : float =0.6):
    # Garante que o datetime está correto
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%y %H:%M:%S')
    qa_pairs = []
    answers = []
    # Filtra apenas perguntas
    questions = df[df['is_question']].copy()
    
    for i, row in questions.iterrows():
        question_time = row['datetime']
        window_end = question_time + timedelta(hours=max_response_hours)
        
        # Filtra respostas potenciais (dentro da janela temporal e de autor diferente)
        potential_answers = df[
            (df['datetime'] > question_time) & 
            (df['datetime'] <= window_end) &
            (df['author'] != row['author']) &
            (~df['is_question'])
        ].copy()
        
        if len(potential_answers) == 0:
            continue
            
        # Se temos modelo de tópicos, calculamos similaridade
        if embeddings_model is not None:
            try:
                # Converte para lista para evitar problemas de indexação
                answer_contents = potential_answers['content'].tolist()
                
                # Gera embeddings (garanta que não há valores nulos)
                question_embedding = embeddings_model.encode(row['content'], convert_to_tensor=True)
                answer_embeddings = embeddings_model.encode(answer_contents, convert_to_tensor=True)
                
                # Calcula similaridade
                similarities = util.pytorch_cos_sim(question_embedding, answer_embeddings)[0].cpu().numpy()

                
                potential_answers['similarity'] = similarities
                answers = potential_answers
                #answers = potential_answers[potential_answers['similarity'] >= similarity_threshold]
            except Exception as e:
                print(f"Erro ao calcular similaridade para pergunta '{row['content']}': {str(e)}")
                answers = potential_answers
        else:
            answers = potential_answers
            
        if len(answers) == 0:
            continue
            
        # Processa as respostas (agora usando .itertuples() que é mais robusto)
        for answer in answers.itertuples():
            qa_pairs.append({
                'question_id': i,
                'question_time': question_time.strftime('%d/%m/%Y %H:%M'),
                'question_author': row['author'],
                'question': row['content'],
                'answer_time': answer.datetime.strftime('%d/%m/%Y %H:%M'),
                'answer_author': answer.author,
                'answer': answer.content,
                'response_time_min': round(
                    (answer.datetime - question_time).total_seconds() / 60, 1
                ),
                'semantic_similarity': getattr(answer, 'similarity', None)
            })
    
    return pd.DataFrame(qa_pairs)

def clean_text(text):
    # Garante que temos uma string
    text = str(text)
    
    # Limpeza básica
    text = re.sub(r'http\S+|@\w+|[^\w\s]|\d+', '', text)
    text = text.lower()
    
    # Tokenização com tratamento de erro
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Se falhar, tenta em inglês como fallback
        tokens = word_tokenize(text)
    
    # Remove stopwords em português
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)


import tiktoken

def cortar_por_token(df: pd.DataFrame, limite_tokens: int = 1048575):
    # Inicializa o tokenizer (modelo da OpenAI, pode mudar conforme seu caso)
    enc = tiktoken.get_encoding("cl100k_base")

    # Garante que o DataFrame está ordenado do mais antigo para o mais recente
    df = df.sort_values(by=["date", "time"])

    # Concatena todas as colunas como uma única string por linha
    df["linha_concat"] = df.apply(lambda row: f"{row['date']} {row['time']} {row['author']} {row['content']} {row['is_question']}", axis=1)

    # Conta tokens por linha
    df["n_tokens"] = df["linha_concat"].apply(lambda x: len(enc.encode(x)))

    # Faz a soma acumulada dos tokens
    df["tokens_acumulados"] = df["n_tokens"].cumsum()

    # Encontra o índice inicial para manter o total abaixo do limite
    idx_inicio = df[df["tokens_acumulados"] > (df["tokens_acumulados"].iloc[-1] - limite_tokens)].index[0]

    # Retorna o DataFrame cortado
    return df.loc[idx_inicio:].drop(columns=["linha_concat", "n_tokens", "tokens_acumulados"])

# ✅ Exemplo de uso:
# df_filtrado = cortar_por_token(seu_dataframe)
