# The four rules, side by side

Pure ASCII throughout, so these paste into a LaTeX `verbatim` block or a plain-text appendix
without escaping. Every panel shows the same three things: what a cell holds, what the update
reads, and what decides the new value.

---

## 1. The update rule

```
  (a) ELEMENTARY CA               (b) DOMANY-KINZEL PCA          
      Wolfram rule 110                p2 = 0 line                
  ---------------------           ---------------------          
  cell state: {0,1}               cell state: {0,1}              
                                                                 
    x     x     x                   x           x                
   i-1    i    i+1                 i-1         i                 
     \    |    /                     \         /                 
      \   |   /                       \       /                  
       +--------+                      +--------+                
       | 8-row  |                      |   p1   |  ONE number    
       | table  |                      +--------+                
       +--------+                          |                     
           |                               v                     
           v                             x'_i                    
         x'_i                                                    
                                                                 
  update: synchronous             update: synchronous            
  free params: 8 bits             free params: 2 reals           
  answer: known exactly           answer: known exactly (DP)     


  (c) THIS PROJECT -- AR rule            (d) THIS PROJECT -- MLM rule
      p(x_i | x_{i-2}, x_{i-1})              p(x_i | x_{i-2..i+2}, centre masked)
      the artifact (F62-F66)                 the clean one (F66)
  ---------------------------            -------------------------------------
  cell state: |V| ~ 50,000               cell state: |V| ~ 30,000
                                                                 
   x     x     x     x     x              x     x     x     x     x
  i-2   i-1    i    i+1   i+2            i-2   i-1    i    i+1   i+2
   +-----+     ^     .     .              +-----+-[MASK]-+-----+
      |        |     :     :                    |     ^
      | LEFT   |   ignored                      | BOTH sides
      | ONLY   |                                |
      v        |                                v
  +-------------+                        +-------------+
  | pythia-410m |                        |  bert-base  |
  |  410M params|                        |  110M params|
  +-------------+                        +-------------+
         |                                      |
         v                                      v
   softmax(logits / T)                    softmax(logits / T)
         |                                      |
         v                                      v
   inverse-CDF sample with u_i            inverse-CDF sample with u_i

  update: ASYNC, random visit order      update: ASYNC, random visit order
  free params: 410,000,000               free params: 110,000,000
  answer: UNKNOWN -- that is the point   answer: UNKNOWN
```

**The one difference that mattered.** Panel (c) hands the model a **two-token prompt**. Pythia was
trained on 2048-token contexts, so this is far outside anything it saw. Panel (d) hands BERT a
masked centre with symmetric context, which **is** its training objective. That single difference
is what F66 isolated: (c) collapses to one filler token at low temperature, (d) never concentrates.

---

## 2. The lattice and the measurement (identical in all four)

```
  a ring of N cells, periodic:

        x_0  x_1  x_2  x_3  ...  x_{N-2}  x_{N-1}
         ^                                   |
         +-----------------------------------+

  DAMAGE SPREADING under common random numbers (CRN):

    replica A:   . . . x   x   x   x   x . . .
                           |
                           | flip ONE cell
                           v
    replica B:   . . . x   x   y   x   x . . .

    both replicas then evolve with THE SAME:
      - model / rule
      - initial state (except the one flipped cell)
      - uniform random stream u
      - visit order            <-- see the F57 note below

    damage(t) = number of cells where A and B differ

         t=0   |....#....|      1 site
         t=1   |...###...|
         t=2   |..#####..|      the "light cone"
         t=3   |.#######.|
          .              .
          .              .
         t=T   |#########|      spread   -- or --
               |.........|      healed   (absorbing state)

  NULL TEST: with NO flip, A and B must stay bit-identical forever.
             Asserted as EXACTLY ZERO differing cells, on every backend.
```

---

## 3. Why the visit order is drawn per replica (F57)

```
  SHARED order (the old default)            PER-REPLICA order (opt-in, F57)
  ----------------------------              -------------------------------
  one permutation per sweep,                each replica draws its own
  used by every replica in the batch

  replica 1:  3 7 1 5 2 ...                 replica 1:  3 7 1 5 2 ...
  replica 2:  3 7 1 5 2 ...   <-- same      replica 2:  6 2 9 1 4 ...
  replica 3:  3 7 1 5 2 ...   <-- same      replica 3:  1 8 3 7 5 ...

  The AR rule is causal-LEFT, so damage at site j spreads only if j+1 or
  j+2 is visited BEFORE j. Otherwise j resamples against an identical
  context with the same u, heals, and the run is absorbed.

  That is 1/3 of orders -- and when the order is shared it kills the
  WHOLE BATCH at once, so 512 "replicas" carry the weight of one draw.

  This was the long-unexplained cause of F42's unignited runs.
```

---

## 4. What each rung buys

| | (a) elementary CA | (b) Domany-Kinzel | (c) AR token CA | (d) MLM token CA |
|---|---|---|---|---|
| cell state | `{0,1}` | `{0,1}` | ~50,000 tokens | ~30,000 tokens |
| window | symmetric, r=1 | 2 parents | **left only, r=2** | **symmetric, r=2, centre masked** |
| rule | 8-row table | one probability | 410M-param network | 110M-param network |
| update | synchronous | synchronous | async, random order | async, random order |
| perturbation | 1 bit | 1 bit | 1 token (O(1) in 50k) | 1 token (O(1) in 30k) |
| ground truth | exact | exact (DP exponents) | none | none |
| role here | validation rung | **the exact rung** -- damage field is provably a DK automaton | the measurement... | ...and its clean counterpart |
| status | F27/F33-F36 | F38, bit-exact, 0 mismatches | **artifact (F62-F66)** | untested for dynamics (#89) |

The ladder principle in one line: **(a) and (b) have known answers, so they calibrate the
estimator before it is pointed at (c) or (d), where nothing is known.** Every retraction in this
project came from skipping or mis-scoping that step.

---

## 5. Where the artifact lives

```
  pythia-410m, AR rule, settled lattice composition vs temperature:

    T        0.02   0.20   0.35   0.40   0.436   0.50   0.60   0.70
    newline%   74     78     70     58      52     34     18     13
    distinct 12.4   10.9   21.0   29.4    34.6   48.5   62.1   66.1
                                     ^                            ^
                                   F58's T_c              the submitted paper

  At T_c the lattice is HALF newlines. The "critical point" is that state
  melting. At T=0.7, where the paper operates, it reads as fragmentary text.

  Interventions that kill it (F65, F66):

    r = 2        74%  <-- the project's window
    r = 4        20%      gone
    r = 8        30%      gone      (the control acquires one at r=16,
    r = 16       55%      BACK <--   so r=16 is a generic long-context effect)

    ban '\n'     74% -> 15%   one token carries all of it, and it does
                              NOT relocate

    + BOS token  74% -> 24%   one token of prefix removes two thirds

    MLM rule     9-14%        never concentrates, at any T or r
```
