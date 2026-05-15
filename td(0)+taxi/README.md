# TD(0) - Taxi-v4

Implementacja jest zgodna z planem cwiczenia:
- `gymnasium`, `numpy`, `matplotlib`,
- srodowisko `Taxi-v4`,
- funkcja uczaca przyjmujaca `alpha`,
- porownanie `alpha = 0.01, 0.1, 0.9`,
- wykres zbieznosci z wygladzeniem,
- dodatkowy baseline i ewaluacja koncowa rozwiazania.

## Krok 1: Inicjalizacja i konfiguracja

W [main.py](C:/Users/barte/PycharmProjects/RL_MwUM/td(0)+taxi/main.py):
- importy: `gymnasium`, `numpy`, `matplotlib.pyplot`,
- srodowisko: `gym.make("Taxi-v4")`,
- stale globalne:
  - `EPISODES = 5000`,
  - `GAMMA = 0.99`,
  - `EPSILON = 0.1`,
  - `ALPHAS = [0.01, 0.1, 0.9]`,
  - `RUNS_PER_ALPHA = 8`.

## Krok 2: Funkcja uczaca

Funkcja `train_td0(alpha, ...)`:
- tworzy tablice `Q` o wymiarach `[n_states, n_actions]`,
- prowadzi petle po epizodach i krokach,
- wybiera akcje polityka epsilon-zachlanna,
- wykonuje krok `env.step(action)`,
- aktualizuje `Q` one-step TD(0) (SARSA(0)):

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
```

W stanie terminalnym target to samo `r`.

Z kazdego epizodu zapisywana jest suma nagrod i funkcja zwraca liste/array tych nagrod.

## Krok 3: Eksperymenty dla roznych alpha

W `main()`:
- dla kazdej wartosci `alpha` uruchamiane jest `RUNS_PER_ALPHA = 8` niezaleznych przebiegow,
- wyniki sa usredniane po przebiegach, zeby ograniczyc szum eksploracji.
- dodatkowo liczony jest baseline: losowa polityka bez uczenia.

## Krok 4: Wykres i analiza

Tworzony jest jeden wykres `td0_taxi_alpha_comparison.png`:
- os X: epizod,
- os Y: suma nagrod,
- trzy linie uczace (`alpha=0.01`, `0.1`, `0.9`) + linia baseline,
- wygladzenie srednia ruchoma (`okno=50`).

Dodatkowo skrypt wypisuje srednia i odchylenie z ostatnich 200 epizodow dla kazdego `alpha`.
Na koncu wypisywana jest tez ewaluacja greedy policy po uczeniu (200 epizodow).

## Oczekiwane obserwacje do prezentacji

- `alpha = 0.01`:
  - uczenie wolne,
  - krzywa rośnie stopniowo i ostroznie.
- `alpha = 0.9`:
  - szybkie i agresywne zmiany,
  - czesto wieksza niestabilnosc (nadpisywanie wiedzy, duze skoki).
- `alpha = 0.1`:
  - zwykle kompromis miedzy szybkoscia i stabilnoscia.
- baseline (losowa polityka):
  - wyraznie nizszy poziom nagrod od wyuczonej polityki.

## Wnioski o doborze hiperparametrow

- Zbyt male `alpha` daje stabilne, ale wolne uczenie.
- Zbyt duze `alpha` przyspiesza reakcje, ale moze destabilizowac zbieznosc.
- W praktyce warto testowac kilka `alpha` i wybierac kompromis na podstawie wykresow i statystyk koncowych.
