# 🤖 Q-Learning i Tajemnica Parametru Alpha (α)

**Temat 5:** Tempo uczenia - strojenie parametru kroku ($\alpha$)  
**Środowisko:** CliffWalking-v1  
**Algorytm:** Q-Learning

Niniejszy dokument stanowi kompletny scenariusz oraz bazę wiedzy do prezentacji (ok. 5-7 minut). Skupiamy się w nim na zrozumieniu mechaniki uczenia ze wzmocnieniem (RL) oraz na tym, dlaczego pojedynczy parametr ($\alpha$) może zadecydować o "geniuszu" lub "głupocie" sztucznej inteligencji.

---

## 🧠 1. O co chodzi w Q-Learningu? (Intuicja i Matematyka)

### Intuicja
Wyobraź sobie, że uczysz psa sztuczek. Gdy pies zrobi coś dobrze, dajesz mu smakołyk (nagrodę). Gdy zrobi coś źle, ignorujesz go lub dostaje karę. Z czasem pies uczy się, przypisując subiektywną "wartość" do konkretnych zachowań w konkretnych sytuacjach. 

Q-Learning robi dokładnie to samo, tworząc ogromną ściągawkę zwaną **Tabelą Q**. Tabela ta przechowuje wartości (Q-values) dla każdej możliwej akcji w każdym możliwym stanie (np. *"jeśli stoję na skraju przepaści, to krok w przód ma wartość -100, a krok w tył +1"*).

### Logika Matematyczna
Sercem algorytmu jest aktualizacja wiedzy na podstawie **Błędu TD** (Temporal Difference Error). 

Agent na bieżąco koryguje swoje przewidywania, patrząc krok w przód. Mówi sobie:  
*"Zrobiłem krok, dostałem nagrodę, a nowe miejsce w którym jestem, wydaje się być warty X. Skoro tak, to moja poprzednia ocena tego ruchu była błędna o Y. Muszę ją poprawić!"*

Równanie Q-Learningu wygląda tak:
$$ Q(S, A) \leftarrow Q(S, A) + \alpha \cdot \Big[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \Big] $$

Gdzie fragment w nawiasach kwadratowych to właśnie nasza **lekcja do przyswojenia** (czyli różnica między nową, lepszą rzeczywistością, a naszą starą wiedzą). I tutaj na scenę wkracza nasz główny bohater – **$\alpha$**.

---

## 🎛️ 2. Parametr Alpha ($\alpha$): Suwak Elastyczności Umysłu

Parametr $\alpha$ (współczynnik uczenia / learning rate) określa precyzyjnie, **jak dużą wagę agent przykłada do nowych doświadczeń w stosunku do starej wiedzy**.

Z matematycznego punktu widzenia po małym przekształceniu wzór ten to zwykła **wykładnicza średnia krocząca**:
$$ Q_{nowe} = (1 - \alpha) \cdot Q_{stare} + \alpha \cdot \text{Nowe Doświadczenie} $$

Jak widać, $\alpha$ to przelicznik wagowy 100% uwagi agenta. Jeśli $\alpha = 0.1$, agent ufa starej wiedzy w 90%, a nowej ufa zaledwie w 10%. Z każdym krokiem nowa informacja powoli, cierpliwie i stabilnie "wygładza" starą.

### 🔄 Co by było, gdybyśmy NIE mieli Alpha? (Odwrotna Logika)

Zadajmy sobie prowokacyjne pytanie: co jeśli w ogóle wyrzucimy $\alpha$ do kosza? Wyobraźmy sobie skrajności matematyczne:

*   **Brak nowej wiedzy ($\alpha = 0$) - "Uparty Ignorant":**  
    Równanie sprowadza się do: $Q_{nowe} = 1 \cdot Q_{stare} + 0$. Agent staje się ślepy na nowe bodźce. Niezależnie od tego, jak fantastyczną i bezpieczną drogę do mety odkryje, jego wrodzona matematyczna ignorancja sprawia, że nie zapamiętuje kompletnie niczego. Uczenie w ogóle nie istnieje.
*   **Całkowity brak pamięci ($\alpha = 1$) - "Syndrom Złotej Rybki":**  
    Równanie to teraz: $Q_{nowe} = 0 \cdot Q_{stare} + 1 \cdot \text{Nowe Doświadczenie}$. Cała przeszłość przestaje mieć znaczenie. Agent wykasowuje tysiące epizodów ciężko zdobytej wiedzy na podstawie jednego ułamka sekundy. Załóżmy, że zna idealną drogę omijającą klif. Raz, przez celową, losową eksplorację, potyka się i spada. Przy $\alpha=1$ natychmiast stwierdza: *"O nie, ta droga to pewna śmierć!"*, wymazując fakt, że poprzednie 999 razy bezpiecznie przeszedł tym szlakiem. 

Dlatego znalezienie "złotego środka" jest być albo nie być dla algorytmu.

---

## 🏞️ 3. Nasz Eksperyment: CliffWalking-v1 i Skrajne Wartości $\alpha$

Aby zademonstrować to zjawisko empirycznie, użyliśmy środowiska `CliffWalking-v1`. Agent musi przejść z punktu startowego do mety, idąc cienką krawędzią tuż nad przepaścią. Spadek to potężna kara (-100). Uruchomiliśmy i narysowaliśmy zachowanie dla 3 wartości $\alpha$:

### 🐢 1. Zbyt niskie Alpha ($\alpha = 0.01$) - "Bardzo Wolne Uczenie"
*   **Obserwacja na wykresie:** Krzywa pnie się w górę niesamowicie powoli. Linia sumy nagród jest niska i niemrawa przez dziesiątki początkowych epizodów. Wykres tempa uczenia spada minimalnie.
*   **Co się dzieje?** Wiedza agenta z każdym krokiem rośnie zaledwie o 1%. Agent wymaga bolesnego wpadnięcia w przepaść setki razy, zanim upewni się na 100%, że faktycznie jest tam niebezpiecznie.
*   **Wniosek:** Algorytm jest co prawda stabilny, ale skrajnie niewydajny czasowo.

### 🌪️ 2. Zbyt wysokie Alpha ($\alpha = 0.9$) - "Zjawisko Nadpisywania Wiedzy"
*   **Obserwacja na wykresie:** Linia ma bardzo silne "załamania" i "szarpnięcia" skierowane w dół, a wariancja wyników jest ogromna.
*   **Co się dzieje?** Kiedy algorytm jest już świetnie wyuczony, incydentalny losowy błąd (np. spowodowany zbadanym ruchem w bok) **nadpisuje** starą, wypracowaną dobrą ocenę stanu wartością z wypadku. Agent gwałtownie "psuje" dobrą mapę wartości i zaczyna błądzić od nowa. 
*   **Wniosek:** Brak ostatecznej, stabilnej zbieżności. Agent nigdy nie osiąga stanu absolutnej pewności swoich działań.

### 🏆 3. Optymalne Alpha ($\alpha \approx 0.1$) - "Zdrowy Balans"
*   **Obserwacja na wykresie:** Krzywe obu wykresów (nagród i kroków) szybko, płynnie i gładko zbiegają do optimum, po czym utrzymują płaską, stabilną linię.
*   **Co się dzieje?** Stara wiedza amortyzuje pojedyncze anomalie i wypadki przy pracy, ale nowe sukcesy w stałym, bezpiecznym tempie poprawiają oceny stanów.

---

## 🎯 4. Główne Wnioski na Prezentację (Take-aways)

1.  **Hiperparametry decydują o wszystkim:** Nawet idealny, genialny pod względem matematycznym algorytm Reinforcement Learningu polegnie i nie rozwiąże problemu, jeśli użyjemy złych hiperparametrów.
2.  **Stochastyczność wymaga gładzenia:** Uczenie ze wzmocnieniem to podróż pełna przypadkowości. Parametr $\alpha$ działa jak amortyzator w aucie – zbyt twardy (1.0) spowoduje, że poczujemy każdy mały kamień, gubiąc trajektorię; zbyt miękki (0.01) sprawi, że w ogóle nie zareagujemy na zmianę ukształtowania terenu.
3.  **Dobór zależy od środowiska:** Nie ma jednej słusznej wartości $\alpha$. Środowiska całkowicie powtarzalne bez elementów losowości mogą dobrze działać przy wyższym Alpha. Ale tam, gdzie panuje szum, niepewność i losowość przejść (jak śliskie kafelki w *FrozenLake* czy eksploracja w *CliffWalking*), uśrednianie wiedzy przy niskim $\alpha$ to jedyna droga do sukcesu.
