import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import wntr
from rag import retriever

load_dotenv()
client = OpenAI(
    base_url = "https://api.groq.com/openai/v1",
    api_key= os.environ["GROQ_API_KEY"],
)
MODELE = "openai/gpt-oss-120b"

# 0. Simulation (une seule fois, au démarrage)
wn = wntr.network.WaterNetworkModel("Net3.inp")
resultats = wntr.sim.EpanetSimulator(wn).run_sim()
pressions = resultats.node["pressure"]


#1. L'outil : une fonction python normale
def pression_noeud(noeud, heure):
    if noeud not in pressions.columns:
        return f"Le noeud {noeud} n'existe pas dans le réseau."
    secondes = heure*3600
    if secondes not in pressions.index:
        return f"Pas de résultat pour l'heure {heure}."
    valeur = pressions.loc[secondes, noeud]
    return f"Pression au noeud {noeud} à {heure}h : {valeur:.2f} m"

def chercher_documentation(question):
    resultats = retriever.invoke(question)
    return "\n\n".join(
        f"[page {r.metadata['page_label']}]\n{r.page_content}" for r in resultats        
    )
#2. La description de l'outil, pour le LLM
outils = [{
    "type" : "function",
    "function" : {
        "name" : "pression_noeud",
        "description" : "Donne la pression (en mètres) à un noeud du réseau d'exemple Net3, à une heure donnée de la simulation.", 
        "parameters" : {
          "type": "object",
          "properties" : {
              "noeud" : {"type" : "string", "description" : "Identifiant du noeud, par exemple '10' "},
              "heure" : {"type" : "integer", "description" : "Heure de la simulation, de 0 à 24"},

          },
          "required" : ["noeud", "heure"], 
        },
    },

},{ 
    "type": "function",
    "function": {
        "name": "chercher_documentation",
        "description": "Cherche des passages dans le manuel utilisateur d'EPANET 2.2 pour répondre aux questions sur le fonctionnement du logiciel, ses concepts et son utilisation.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "La question ou les mots-clés à chercher, de préférence en anglais car le manuel est en anglais"},
            },
            "required": ["question"],
        },
    },
    
}]

#3. L'agent
fonctions = {
    "pression_noeud": pression_noeud,
    "chercher_documentation": chercher_documentation,
}

def agent(question, max_tours=5):
    messages = [
        {"role": "system", "content": (
            "Tu es l'assistant EPANET, un prototype développé pour aider les utilisateurs "
            "du logiciel de simulation hydraulique EPANET. "
            "Tu as deux outils : chercher_documentation pour les questions sur EPANET, "
            "et pression_noeud pour les pressions du réseau Net3. "
            "Pour les questions sur EPANET, réponds uniquement à partir de la documentation trouvée ; "
            "si elle ne contient pas la réponse, dis-le. "
            "Ne fais pas plus de deux recherches pour une même information : si tu ne trouves pas, dis-le et continue. "
            "Cite les pages utilisées sous la forme (p. 3), en t'appuyant sur les numéros [page X] du contexte. "
            "N'invente jamais de valeurs numériques : utilise toujours un outil pour les obtenir. "
            "Réponds dans la même langue que la question."
                        "Pour une valeur obtenue par simulation, n'ajoute aucune référence de page. "
        )},
        {"role": "user", "content": question},
    ]

    for tour in range(max_tours):
        # Le LLM réfléchit : réponse finale ou demande d'outil ?
        response = client.chat.completions.create(
            model=MODELE, messages=messages, tools=outils, max_tokens=1500
        )
        message = response.choices[0].message

        # Plus besoin d'outil -> c'est la réponse finale
        if not message.tool_calls:
            return message.content

        # Sinon, on garde sa demande dans l'historique
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": a.id, "type": "function",
                 "function": {"name": a.function.name, "arguments": a.function.arguments}}
                for a in message.tool_calls
            ],
        })

        # On exécute chaque outil demandé et on renvoie le résultat
        for appel in message.tool_calls:
            arguments = json.loads(appel.function.arguments)
            print(f"🔧 [tour {tour + 1}] Outil appelé :", appel.function.name, arguments)
            resultat = fonctions[appel.function.name](**arguments)
            messages.append({"role": "tool", "tool_call_id": appel.id, "content": resultat})

    return "Désolé, je n'ai pas pu terminer la réponse (trop d'étapes)."

#4. Tester
if __name__ == "__main__":
    for q in [
        "À quoi sert EPANET ?",
        "Quelle est la pression au nœud 10 à 6h ?",
        "Qu'est-ce qu'une jonction dans EPANET, et quelle est la pression au nœud 15 à 12h ?",
        "Bonjour, qui es-tu ?",
    ]:
        print("\n=== Question :", q)
        print(agent(q))