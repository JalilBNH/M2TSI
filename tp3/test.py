import tensorflow as tf

# Vérifier si les GPU sont détectés
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs détectés : {len(gpus)}")
else:
    print("Aucun GPU détecté")