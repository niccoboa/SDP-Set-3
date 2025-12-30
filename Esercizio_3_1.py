### Autore: Niccolò Boanini (7197315) - Dicembre 2025 ###
### Set di Esercizi 3: Test di Ipotesi (3.1) ###
### Codice per eseguire il test di ipotesi basato sull'Indice di Coincidenza ###

import random
import math
import re
from collections import Counter
from scipy.stats import norm

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# --- FUNZIONI DI SUPPORTO ---

def load_text(filename):
    """
    Carica il testo del libro e lo pulisce (solo a-z).
    """
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    text = re.sub(r'[^a-z]', '', text)  # Rimuove tutto tranne a-z
    return text


def random_text(n):
    """Genera testo casuale (Ipotesi H0)"""
    text = ""
    for i in range(n):
        char = random.choice(ALPHABET)
        text = text + char
    return text

def english_text_iid(n):
    """
    Genera Inglese 'Sintetico' (I.I.D.).
    Usa le frequenze corrette, ma estrae lettere a caso (nessuna regola grammaticale reale!).
    """
    text = ""
    chars = random.choices( # estrazione con rimpiazzo secondo le frequenze della tabella
        population=ALPHABET, 
        weights=list(LETTER_FREQUENCY_TABLE.values()), 
        k=n
    )

    for char in chars:
        text = text + char
    return text

def english_text_real(n):
    """
    Estrae una sottostringa REALE di lunghezza n da un testo.
    Simula H1 Reale (ad es. in questo caso con il primo capitolo di Moby Dick).
    """
    # si assume che il testo sia sufficientemente lungo (perlomeno >n)
    start = random.randint(0, len(PLAINTEXT) - n)
    end = start + n
    return PLAINTEXT[start:end]

def random_shift_cipher(plain_text):
    """Applica uno shift casuale (Cifrario di Cesare)"""
    shift = random.randint(1, 25)
    cipher_text = ""
    for char in plain_text:
        index = ALPHABET.index(char)
        new_index = (index + shift) % 26
        cipher_text += ALPHABET[new_index]
    return cipher_text

# --- FUNZIONI MATEMATICHE E TEST ---

def ic(text):
    """Calcola l'Indice di Coincidenza del testo."""
    n = len(text)
    if n < 2: return 0
    counts = Counter(text) # conta le occorrenze di ogni carattere
    num = 0
    for f in counts.values():
        num += f * (f - 1)
    den = n * (n - 1)
    return num / den

def hp_test(text, threshold):
    """
    Test di Ipotesi: Ritorna 1 se Inglese, 0 se Random.
    Soglia = 0.052
    """
    ic_value = ic(text)
    if ic_value > threshold:
        return 1 # H1
    else:
        return 0 # H0

def phi(z):
    """
    Calcola la CDF della Normale Standard Phi(z).
    """
    # alternativa: return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return norm.cdf(z)

def normal_cdf(x, mu, sigma):
    """
    Calcola P(X <= x) standardizzando prima la variabile.
    Ovvero Phi(z) con z = (x - mu) / sigma.
    """
    z = (x - mu) / sigma #Standardizzazione
    
    return phi(z) # Funzione Phi(z) (CDF della Normale Standard)

def get_theorical_errors(N, threshold):
    """Calcola alpha e beta (teorici, "precisi")."""
    # H0: Random
    mu0 = 0.03846 # approx 1.0 / 26.0
    sigma0 = math.sqrt(2 * (26 - 1)) / (26 * math.sqrt(N * (N - 1)))
    alpha = 1 - normal_cdf(threshold, mu0, sigma0)
    
    # H1: Inglese (IID)
    mu1 = 0.065
    sigma1 = sigma0  # ipotesi: gaussiana con stessa varianza
    beta = normal_cdf(threshold, mu1, sigma1)

    print_overlapped_gaussians(mu0, sigma0, mu1, sigma1, threshold, N)
    
    return alpha, beta

def print_overlapped_gaussians(mu0, sigma0, mu1, sigma1, threshold, N):
    x = np.linspace(mu0 - 3*sigma0, mu1 + 3*sigma0, 100)
    
    # Grafico delle due gaussiane
    plt.figure(figsize=(10, 6))
    plt.plot(x, stats.norm.pdf(x, mu0, sigma0), label='H0: Random', color='blue')
    plt.plot(x, stats.norm.pdf(x, mu1, sigma1), label='H1: Inglese', color='red')
    
    # Linee verticali per soglia e medie
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Soglia T={threshold}')
    plt.axvline(x=mu0, color='blue', linestyle='--', linewidth=1, label=f'μ0={mu0:.3f}')
    plt.axvline(x=mu1, color='red', linestyle='--', linewidth=1, label=f'μ1={mu1:.3f}')
    
    # Aree di errore alpha e beta
    x_alpha = np.linspace(threshold, x[-1], 200)
    plt.fill_between(x_alpha, stats.norm.pdf(x_alpha, mu0, sigma0), color='blue', alpha=0.2, label=r'Errore $\alpha$')
    x_beta = np.linspace(x[0], threshold, 200)
    plt.fill_between(x_beta, stats.norm.pdf(x_beta, mu1, sigma1), color='red', alpha=0.2, label=r'Errore $\beta$')
    
    # Dettagli del grafico (titoli, etichette, legenda)
    plt.title(f'Distribuzioni di Probabilità IC per N={N}')
    plt.xlabel('Indice di Coincidenza (IC)')
    plt.ylabel('Densità di Probabilità')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

    return 1


# --- FUNZIONE PRINCIPALE: TRIPLO CONFRONTO (Random, I.I.D., Reale) ---

def evaluate_test(blocks, n, threshold):
    
    # Contatori degli errori
    fp = 0       # Counter di falsi Positivi (random classificato come testo inglese)
    fn_iid = 0   # Counter di falsi Negativi IID (testo 'sintetico' generato secondo le frequenze inglesi classificato come testo random)
    fn_real = 0  # Counter di falsi Negativi 'REALI' (testo reale [Moby Dick] classificato come testo random)
    
    hblocks = blocks // 2 # cioè half_blocks (la metà)
    print(f"\n--- ESECUZIONE TEST (N={n}, Blocchi={blocks}), soglia={threshold} ---")
    
    # 1. TEST H0 (Random Text) -> Calcola Alpha
    for i in range(hblocks):
        text = random_text(n) # genera testo random
        if hp_test(text, threshold) == 1: # se viene classificato come inglese dal test
            fp += 1
    print("un testo random:", text)
     
    # 2. TEST H1 I.I.D. (Sintetico) -> Calcola Beta (I.I.D.)
    for j in range(hblocks):
        plain_text = english_text_iid(n)
        cipher_text = random_shift_cipher(plain_text)
        if hp_test(cipher_text, threshold) == 0: 
            fn_iid += 1
    print("un testo iid:   ", plain_text, " (da 'shift'-cifrare)")

    # 3. TEST H1 REALE (Moby Dick) -> Calcola Beta (Reale)
    for k in range(hblocks):
        plain_text = english_text_real(n) # genera testo reale (es. preso dal primo capitolo di Moby Dick)
        cipher_text = random_shift_cipher(plain_text) # cifratura shift
        if hp_test(cipher_text, threshold) == 0: # se viene classificato come random
            fn_real += 1
    print("un testo reale: ", plain_text, " (da 'shift'-cifrare)")


    # CALCOLO degli ERRORI
    # Calcolo percentuali errori teorici, "precisi" (formule)
    alpha, beta = get_theorical_errors(n, threshold)

    # Calcolo percentuali errori simulati, "empirici" (sperimentali)
    alpha_emp = fp / hblocks
    beta_iid = fn_iid / hblocks
    beta_real = fn_real / hblocks
    
    # Tabella risultati (confronto)
    print(f"{'TIPO DI ERRORE':<25} | {'VALORE':<10}")
    print("-" * 40)
    print(f"{'Alpha Teorico':<25} | {alpha:.2%}")
    print(f"{'Alpha Empirico':<25} | {alpha_emp:.2%}")
    print("-" * 40)
    print(f"{'Beta Teorico':<25} | {beta:.2%}")
    print(f"{'Beta I.I.D. (Sintetico)':<25} | {beta_iid:.2%}")
    print(f"{'Beta REALE (Moby Dick)':<25} | {beta_real:.2%}")
    print("-" * 40)


# --- SETUP ---

# fonte: https://web.archive.org/web/20080708193159/http://pages.central.edu/emp/LintonT/classes/spring01/cryptography/letterfreq.html
LETTER_FREQUENCY_TABLE = {
    'a': 0.08167, 'b': 0.01492, 'c': 0.02782, 'd': 0.04253, 'e': 0.12702,
    'f': 0.02228, 'g': 0.02015, 'h': 0.06094, 'i': 0.06966, 'j': 0.00153,
    'k': 0.00772, 'l': 0.04025, 'm': 0.02406, 'n': 0.06749, 'o': 0.07507,
    'p': 0.01929, 'q': 0.00095, 'r': 0.05987, 's': 0.06327, 't': 0.09056,
    'u': 0.02758, 'v': 0.00978, 'w': 0.02360, 'x': 0.00150, 'y': 0.01974,
    'z': 0.00074
}
ALPHABET = list(LETTER_FREQUENCY_TABLE.keys())
PLAINTEXT = load_text("chapter01.txt")


# --- AVVIO del test ---
evaluate_test(blocks=100, n=60, threshold=0.05173)