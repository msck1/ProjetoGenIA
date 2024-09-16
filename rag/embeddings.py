from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def main(): # compara embeddings
    embedding_function = OpenAIEmbeddings() # modelo ada 002 que faz o embedding
    vector = embedding_function.embed_query("apple")
    print(f"Vetor para apple: {vector}")
    print(f"Tamanho do vetor: {len(vector)}")

    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1]) # x recebe o valor da funcao evaluate_string_pairs (nesse caso apple e iphone)
    print(f"Comparing ({words[0]}, {words[1]}): {x}") # comparacao dos embeddings é imprimida


if __name__ == "__main__":
    main()