import argparse 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = r""

PROMPT_TEMPLATE = """

Responda a pergunta se baseando apenas no contexto a seguir:

{context}

---


Responda a pergunta se baseando apenas no contexto acima: {question}

"""

def main():
    parser = argparse.ArgumentParser() # cria um objeto parser
    parser.add_argument("query_text", type=str, help="The query text.") # atribui query_text ao parser
    args = parser.parse_args() # processa o query_text via linha de comando (nesse caso o terminal)
    query_text = args.query_text # extrai o texto 

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=5) # busca os valores mais prox de acordo com k
    if len(results) == 0 or results[0][1] < 0.7: # se a relevancia do primeiro resultado for menor que 0.7 termina o programa
        print(f"Não houve resultados")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) # formata o texto dos documentos similares
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text) # formata o prompt com o input do usuario e texto dos documentos
    print(prompt)

    model = ChatOpenAI( 
        model="gpt-4o"
    )
    response_text = model.invoke(prompt)


    sources = [doc.metadata.get("source", None) for doc, _score in results] # extrai os metadados das fontes
    formatted_response = f"Resposta: {response_text}\nSources: {sources}"
    print(formatted_response) # imprime a resposta formatada 


if __name__ == "__main__":
    main()
