### Autore: Niccolò Boanini (7197315) - Dicembre 2025 ###
### Set di Esercizi 3: Sicurezza incondizionata e unicity distance (2.5) ###
### Codice per i plot riportati in Figura 2 della relazione ###

import matplotlib.pyplot as plt
import numpy as np

# Parametri (Dimensione della chiave da 1 a 30)
x = np.arange(1, 26)

# Formule derivate nell'esercizio
y_subst = [27.6] * len(x)        # Costante
y_vig = 1.47 * x                 # Lineare (Vigenère)
y_hill = 1.47 * (x**2)           # Quadratico (Hill)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_subst, label='Sostituzione Monoalfabetica (Costante)', color='red', linestyle='--')
plt.plot(x, y_vig, label='Vigenère (Lineare)', color='blue')
plt.plot(x, y_hill, label='Hill (Quadratico)', color='green')


plt.title('Confronto Unicity Distance ($n_0$)')
plt.xlabel('Parametro di complessità (Lunghezza chiave L, oppure Dimensione matrice m)')
plt.ylabel('Caratteri necessari per la rottura del cifrario ($n_0$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 150) 

plt.show()