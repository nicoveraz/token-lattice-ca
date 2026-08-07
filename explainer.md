# What this project is, in plain English

No background assumed. Jargon is defined the first time it appears. If you only read one section,
read §7 — it is the part that turned out to matter.

---

## 1. The one-sentence version

We turned a language model into a simple physics-style system, so we could measure it the way
physicists measure materials — and then spent most of our effort finding out which of those
measurements were real.

---

## 2. What a cellular automaton is

Imagine a long row of boxes. Each box holds a value — say, black or white. There is one rule that
says: *look at a box and its immediate neighbours, and that tells you what the box becomes next.*
Apply the rule everywhere, over and over.

That's a **cellular automaton** (CA). It is one of the simplest systems that can still do
complicated things. From a trivial rule and a row of boxes, you can get patterns that look random,
patterns that settle into stripes, and patterns that do neither.
<>
The reason people care: CAs sit at the boundary between "so simple it's boring" and "so complex
it's unanalysable". They are a standard laboratory for asking *when does local, simple behaviour
produce global, complicated behaviour?*

Two properties matter for what follows:

- **A CA can freeze or churn.** Some rules settle into a fixed pattern and stay there. Others never
  settle. Some sit at a boundary between the two.
- **You can poke it.** Change one box and re-run. Does the change stay put, die out, or spread
  across the whole row? That question has a name — **damage spreading** — and it is the main tool
  in this project.

---

## 3. Turning a language model into one

Now replace "black or white" with "a word".

Take a ring of, say, 96 slots. Each slot holds one token — roughly a word or word-piece. That's our
row of boxes, bent into a circle so there are no edges.

For the rule, we use a language model. Pick a slot. Show the model the neighbouring tokens. Ask it:
*what token goes here?* It answers with a probability for every token in its vocabulary — about
50,000 of them. We pick one according to those probabilities and write it into the slot. Then we
move to another slot and do it again, in random order, forever.

That's it. That's the whole construction. A language model is now a rule for a cellular automaton
over text.

Two dials:

- **Radius (r)** — how many neighbours the model gets to see. We mostly used **r = 2**, meaning two
  tokens.
- **Temperature (T)** — how adventurous the sampling is. Low temperature means the model almost
  always picks its top choice. High temperature means it takes more risks.

**Why do this at all?** Because it lets you ask questions about a language model that you cannot
otherwise ask. Language models are usually studied by prompting them and reading the output. Here
we get a *dynamical system* — something that evolves in time, that can be perturbed, that can have
phases and transitions. All of the machinery physicists built for magnets and epidemics becomes
available.

---

## 4. How you measure anything in it

The core measurement is damage spreading, and it works like this.

Run two identical copies of the system side by side. Give them the same starting text, the same
random numbers, the same order of slot visits — everything identical. They will stay identical
forever. That's the **null test**, and we check it obsessively: the two copies must differ in
**exactly zero** slots. Not "almost zero". Zero. If that ever fails, every number downstream is
meaningless.

Now change **one token** in one copy. That's the poke.

```
    copy A:   . . . the  cat  sat  on  the . . .
                          |
                          | change one token
                          v
    copy B:   . . . the  cat  ran  on  the . . .

    then let both run, and count the slots where they differ:

         t=0   |....#....|      1 slot
         t=1   |...###...|
         t=2   |..#####..|      spreading
         t=3   |.#######.|
          .
         t=T   |#########|   spread everywhere
                    or
               |.........|   healed completely
```

If the difference **heals**, the system is in a stable, "frozen" state — it forgets disturbances.
If it **spreads**, the system is chaotic — small changes take over. In between, at some particular
temperature, there can be a sharp boundary: a **phase transition**, like water freezing.

Finding that boundary, and measuring its properties precisely, was the goal of the last stretch of
work.

---

## 5. Why we calibrate before we measure

Here is the problem with measuring a language model this way: **nobody knows the right answer**. If
our method returns "the transition has property X", there is nothing to check it against.

So the project's organising rule is: **before measuring something unknown, reproduce something
known.**

We run the exact same code on systems whose answers are already established:

- **Elementary cellular automata** — the classic black-and-white rules. Their behaviour classes are
  documented. Can our tools tell the ordered ones from the chaotic ones? Yes.
- **The Domany–Kinzel automaton** — a simple probabilistic CA that has been studied for forty years.
  The exact temperature where it flips from freezing to spreading is published, and so are its
  **exponents**: a handful of numbers describing *how* it behaves near that flip — how fast the
  damage grows, how long it survives. Exponents are the fingerprint of a transition. Two systems
  that look nothing alike can share them, and when they do, physicists call them the same
  "universality class".

The Domany–Kinzel rung is the strongest, because of a mathematical fact: for this system, the
*difference* between two poked copies is itself provably another copy of the same automaton. So we
have an exact prediction, not an approximate one. Our code reproduces it **bit for bit — zero
mismatching cells, no error bar** — through the very same machinery that produces the
language-model numbers.

This sounds like bureaucracy. It is the single most important thing in the project, and §7 explains
why.

---

## 6. What we found, in order

Briefly, because §7 supersedes most of it:

We found what looked like a genuine phase transition in the token system. Two independent measures
of it agreed on the same temperature — around T ≈ 0.436 — which is exactly what a real critical
point looks like. We measured its exponents and started comparing them against the
default expectation — a class called **directed percolation**, which describes a huge range of
systems where something either dies out or takes over: epidemics, forest fires, fluid seeping
through rock. If our transition matched it, that would place the language model's behaviour in
company with all of those.

Along the way, six confident conclusions turned out to be wrong. Every one was caught by the
calibration step, before it reached a paper:

- A tolerance measured on a small system and wrongly applied to a large one — a threshold so tight
  it rejected the *known-correct* system.
- Error bars computed as if 512 measurements were independent when they were secretly one.
- A curve-fitting procedure that could cheat by shrinking its own comparison window.
- A test that returns the same answer for both possibilities, so it can't distinguish them.
- A result that looked clean until we ran a control that should have shown nothing — and it showed
  the same thing.
- Nineteen different models that couldn't tell us what we needed, until we changed the *setup*
  rather than the model.

---

## 7. The twist: we were measuring our own apparatus

Here is what the transition actually was.

At low temperature, we looked at what the ring of tokens had settled into. For the model we'd been
studying, it was this:

```
    \n \n \n , , , , , , \n \n \n \n \n \n \n \n \n \n \n \n \n ...
```

Newlines. Eighty-one of the ninety-six slots were newline characters. The "frozen, stable phase"
was not the system settling into meaningful text. It was the system collapsing into blank lines.

That is why pokes healed: every slot was going to become a newline no matter what you did to its
neighbours.

Three experiments pinned down the cause:

1. **Only at radius 2.** Widen the window from two tokens to four, and the effect vanishes.
2. **One token carries all of it.** Forbid the model from emitting a newline, and the effect
   collapses immediately — and does *not* move to some other token.
3. **A single extra token of context removes two thirds of it.** Prepend one marker saying "this is
   the start of a document", and the collapse mostly stops.

Put together, the explanation is simple. **We were asking the model to continue from two tokens.**
It was trained on contexts of two *thousand*. A two-token prompt is nothing like anything it ever
saw, and when a language model is handed almost no context, it falls back on the most common filler
in its training data. For this model, that's a newline.

The clincher: we ran the same measurement using a *different* kind of model — one trained to fill
in a blank with context on both sides, which is exactly what our setup asks of it. **No collapse at
all**, at any temperature or window size.

So the "phase transition" was the melting of an artifact. It was a real, reproducible, precisely
measurable property — of our probe, not of language models.

---

## 8. What survives

Quite a lot, and the useful parts are not the ones we set out to get.

**A cautionary result with teeth.** People do probe language models by feeding their own output
back in. This project shows, across nineteen models and with the mechanism isolated by three
independent interventions, how that can manufacture a phenomenon that belongs to the probe. That is
worth knowing, and it is not obvious in advance — the transition looked real by every internal
check we had.

**A method for not fooling yourself.** The discipline that caught all six errors generalises:
calibrate at the same settings you'll measure at, not convenient ones; state what your independent
unit of measurement actually is and test it; make a fitted procedure prove it can recover a known
answer; **run a control that should show nothing**; and **vary your setup, not just your subject**.
That last one is the sharpest lesson here — nineteen different models could not distinguish "a fact
about language models" from "a fact about our apparatus", and one change of apparatus did it in an
afternoon.

**An explanation for an old puzzle.** A long-standing oddity in the data — a third of runs
mysteriously producing nothing — turned out to have an exact cause in how the slots were being
visited. It is now understood and fixed.

**A clean starting point.** The alternative setup, the one that doesn't collapse, has barely been
explored. Whether *it* has a phase transition is being measured now — and a "no" would be a good
answer, because it would confirm the transition was only ever the artifact.

**And the submitted paper is untouched.** It operates at a higher temperature, where the ring
contains varied, text-like content rather than blank lines, and it uses a more robust poke. The
artifact lives somewhere the paper never went.

---

## 9. If you want the detail

- **[`ca_constructions.md`](ca_constructions.md)** — the four rules drawn side by side in ASCII.
- **[`findings.md`](findings.md)** — the evidence ledger. Every finding, including every retraction,
  with the numbers.
- **[`README.md`](README.md)** — how to run any of it.
- **[`paper_arxiv/plan_paper2.md`](paper_arxiv/plan_paper2.md)** — what is worth writing up, and what isn't.

One note on reading the ledger: it contains a lot of retracted claims. That is deliberate. A
retraction that stays visible is evidence the process works; a retraction that quietly disappears
is evidence it doesn't.
