# Open points

- WebNLG-IT contains only English triples, whereas the Spanish dataset also includes Spanish triples. How should we handle this difference? Here are two possible options:
    - We can extract the Italian translation of the triples (perhaps using the same strategy Virginia adopted in her previous work on the Spanish WebNLG).
      _--> Virginia: Might be too tight of a timeline to extract and validate the italian translation too, maybe stick only to English for this one._
    - We can use only the English triples in the experiments (e.g., “Verbalized in Italian, this triple english_triple → Italian verbalization”).

    **choice**: Use only the English triples in the experiments.

- Are the examples provided in the one/few-shot settings taken from the LongTailWebNLG dataset? If so, I think we need to exclude them from the experiments.
    **choice**: Yes, they are taken from the LongTailWebNLG dataset. We will exclude them from the experiments.

- There are some errors in the LongTailWebNLG dataset that need to be corrected.
  **actions**
  - WKQ6IC --> Corretta (rimosso la nota)
  - 3HHTM0 --> Corretto il soggetto per l'italiano (da Media Vida a Halflife)
  - J8K4KF --> Corretto l'oggetto della seconda tripla (da Maduro lampreado a Quito)
  - T5V4UJ --> Rimossa
