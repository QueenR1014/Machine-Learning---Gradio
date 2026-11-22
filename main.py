from training.training import train_all_models, modelos_listos, cargar_resultados

if not modelos_listos():
    metricas = train_all_models()
    print(metricas["clasificacion"].to_string(index=False))
    print(metricas["regresion"].to_string(index=False))

R = cargar_resultados()
print("Modelos cargados:", list(R["clasificacion"]["modelos"]), list(R["regresion"]["modelos"]))