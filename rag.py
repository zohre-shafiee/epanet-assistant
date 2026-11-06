import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1.Charger le pdf
loader = PyPDFLoader("epanet_manuel.pdf")
pages = loader.load()

# 2.Tirer les chapitres 1 et 2 (pages 14 à 27)
docs = [p for p in pages if p.metadata["page"] >= 14]
print("Pages gardées : ", len(docs))

# 3.Découper en chunk
splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
chunks = splitter.split_documents(docs)
print("Nombre de chunks :", len(chunks))

# 4.Modèle e'embedding multilingue
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 5.Base Vectorielle (en mémoire pour l'instant)
vectorstore = Chroma.from_documents(chunks, embeddings)

# 6.Retriever : renvoie les 3 chunks les plus proches
retriever = vectorstore.as_retriever(search_kwargs= {"k": 3})

# 7. Connexion au LLM
load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
MODELE = "openai/gpt-oss-120b"

# 8.Fonction RAG complète
def repondre(question):
    #R : retrouver les chunks pertinents
    resultats = retriever.invoke(question)

    #A : construire le contexte avec le numéro de page
    contexte = "\n\n".join(
        f"[page {r.metadata['page_label']}]\n{r.page_content}" for r in resultats
    )

    prompt = f"""Tu es un assistant qui aide les utilisateurs du logiciel EPANET.
Réponds uniquement à partir du contexte ci-dessous.
Si la réponse n'est pas dans le contexte, réponds : "Je ne sais pas, cette information n'est pas dans la documentation."
réponds dans la MÊME langue que la question. Si la question est en anglais, réponds en anglais. Si elle est en français, réponds en français.

context : 
{contexte}

Question : {question} """

    #G : Gérer la réponse
    reponse = client.chat.completions.create(
        model = MODELE,
        messages = [{"role": "user", "content": prompt}],
        max_tokens= 1500,
    )
    texte = reponse.choices[0].message.content   
    if texte.startswith("Je ne sais pas"):
        pages = []
    else:
        pages = sorted(set(r.metadata["page_label"] for r in resultats), key=int)
    return texte, pages 

# 9. Tester 
if __name__ == "__main__":
    questions = [
        "À quoi sert EPANET ?",
        "What are the steps in using EPANET?",
        "Quelle est la capitale de la France ?",
    ]

    for q in questions:
        texte, pages = repondre(q)
        print("\n=== Question :", q)
        print(texte)
        if pages:
            print("Sources : pages", ", ".join(pages))