import gradio as gr
from agent import agent
import re

def chat(message, history):
    texte = agent(message)
    texte = re.sub(r"【page (\d+)】", r"(p. \1)", texte)   # 【page 143】 → (p. 143)
    texte = re.sub(r"【[^】]*】", "", texte)                # supprime les autres 【...】
    return texte

demo = gr.ChatInterface(
    fn = chat,
    title="Assistant EPANET : agent IA (documentation + simulation)",
    description="Je peux répondre à vos questions sur le manuel EPANET 2.2 et calculer les pressions du réseau d'exemple Net3, en français ou en anglais.",
    examples=[
        "À quoi sert EPANET ?",
        "Quelle est la pression au nœud 10 à 6h ?",
        "Qu'est-ce qu'une jonction, et quelle est la pression au nœud 15 à 12h ?",
    ],
)


demo.launch()
