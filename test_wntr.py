import wntr 

# 1. Charger le réseau
wn = wntr.network.WaterNetworkModel("Net3.inp")
print("Nombre de noeuds : ", wn.num_nodes)
print("Nombre de tuyaux : ", wn.num_pipes)
print("Nombre de jonctions : ", wn.junction_name_list[:5])

# 2. Lancer la simulation hydraulique
sim = wntr.sim.EpanetSimulator(wn)
resultats = sim.run_sim()

# 3. Lire la pression d'un noeud au début de la simulation
pressions = resultats.node["pressure"]
noeud = wn.junction_name_list[0]
print(f"Pression au noeud {noeud} (t=0) : ", round(pressions[noeud].iloc[0], 2), "m")

