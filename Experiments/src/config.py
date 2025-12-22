"""Configurazione centrale per i path del progetto"""
from pathlib import Path

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODELS = [
    ## TEST
    #'ibm-granite/granite-4.0-micro',
    #"google/gemma-3-1b-it",
    #"meta-llama/Llama-3.2-1B-Instruct"

    ## EXPERIMENTAL
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
]

USE_QUANTIZATION = False

# Generation parameters
INFERENCE_BATCH_SIZE = 8  # Batch size for inference
NUM_GENERATIONS_PER_TRIPLE = 3  # Number of generations per triple
GENERATION_TEMPERATURE = 0.7  # Temperature for generation (>0 for variability)

# ID of one shot and few shot example to exclude:
EXCLUDED_TRIPLES = ["6ILWWO", "JY22GL", "5VTQU4", "O6RKWV", "3Z967Q", "GCHYVN", "9XQ58K", "BV8MZ9", "G4BFIK"]

# Examples for one-shot learning (by language)
ONE_SHOT_EXAMPLES = {
    "en": (
            [
                "[subject: 'Ligia Petit', predicate: 'place of birth', object: 'Maracay']",
                "[subject: 'Ligia Petit', predicate: 'occupation', object: 'model']",
                "[subject: 'Maracay', predicate: 'twinned administrative body', object: 'Salta']",
                "[subject: 'model', predicate: 'partially coincident with', object: 'fashion model']",
                "[subject: 'Maracay', predicate: 'located in the administrative territorial entity', object: 'Municipio Girardot']"
            ],
        "Liga Petit was born in Maracay and works as a model. Maracay is twinned with Salta and is located in Municipio Giradot. The profession of a model partly coincides with that of a fashion model."
    ),
    "es": (
            [
                "[subject: 'Johann Reif', predicate: 'lugar de fallecimiento', object: 'Kritzendorf']",
                "[subject: 'Johann Reif', predicate: 'miembro del partido político', object: 'Partido Socialdemócrata de Austria']",
                "[subject: 'Kritzendorf', predicate: 'se encuentra en el huso horario', object: 'UTC+02:00']",
                "[subject: 'Partido Socialdemócrata de Austria', predicate: 'país', object: 'Austria']",
                "[subject: 'Kritzendorf', predicate: 'situado en la entidad territorial administrativa', object: 'Klosterneuburg']"
            ],
        "Johann Reif falleció en Kritzendorf y fue miembro del Partido Socialdemócrata de Austria. Kritzendorf se encuentra en el huso UTC+02:00 y está situado en la entidad territorial administrativa Klosterneuburg. El país del Partido Socialdemócrata de Austria es Austria"
    ),
    "it": (
            [
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'paese', object: 'Austria']",
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'unità amministrativa in cui è situato', object: 'Vienna']",
                "[subject: 'Austria', predicate: 'organo esecutivo', object: 'Governo federale dell'Austria']",
                "[subject: 'Austria', predicate: 'situato sul mare / lago / fiume', object: 'Morava']",
                "[subject: 'Austria', predicate: 'organo legislativo', object: 'Assemblea Federale (Austria)']"
            ],
        "Il Monumento ai combattenti ebrei si trova in Austria, presso l'unità amministrativa di Vienna. L'Austria, attraverso cui scorre il fiume Morava, ha come organo esecutivo il governo federale dell'austria e come organo legislativo l'assemblea Federale"
    )
}

# Examples for few-shot learning (by language)
FEW_SHOT_EXAMPLES = {
    "en": [
        (
            [
                "[subject: 'Kewpie doll', predicate: 'discoverer or inventor', object: 'Rose O'Neill']",
                "[subject: 'Kewpie doll', predicate: 'creator', object: 'Rose O'Neill']",
                "[subject: 'Kewpie doll', predicate: 'penciller', object: 'Rose O'Neill']"
            ],
            "Kwepie doll was invented, created, and drawed by Rose O'Neill"
        ),
        (
            [
                "[subject: 'Polsterberg Pumphouse', predicate: 'country', object: 'Germany']",
                "[subject: 'Polsterberg Pumphouse', predicate: 'located in the administrative territorial entity', object: 'Clausthal-Zellerfeld']",
                "[subject: 'Germany', predicate: 'head of state', object: 'Frank-Walter Steinmeier']",
                "[subject: 'Germany', predicate: 'shares border with', object: 'Belgium']"
            ],
            "The Polsterberg Pumphouse is located in Germany and specifically in Clausthal-Zellerfeld. Germany borders with Belgium and its president is Frank-Walter Steinmeier"
        ),
        (
            [
                "[subject: 'Ligia Petit', predicate: 'place of birth', object: 'Maracay']",
                "[subject: 'Ligia Petit', predicate: 'occupation', object: 'model']",
                "[subject: 'Maracay', predicate: 'twinned administrative body', object: 'Salta']",
                "[subject: 'model', predicate: 'partially coincident with', object: 'fashion model']",
                "[subject: 'Maracay', predicate: 'located in the administrative territorial entity', object: 'Municipio Girardot']"
            ],
            "Liga Petit was born in Maracay and works as a model. Maracay is twinned with Salta and is located in Municipio Giradot. The profession of a model partly coincides with that of a fashion model"
        )
    ],
    "es": [
        (
            [
                "[subject: 'Alois Pupp', predicate: 'país de nacionalidad', object: 'Italia']",
                "[subject: 'Alois Pupp', predicate: 'lugar de fallecimiento', object: 'Novoli']", 
                "[subject: 'Alois Pupp', predicate: 'cargo ocupado', object: 'Florencia']"
            ],
            "Alois Pupp es de nacionalidad italiana, falleció en Bresanona y ocupó el cargo de presidente del Tirol del Sur."
        ),
        (
            [
                "[subject: 'Dev Gill', predicate: 'país de nacionalidad', object: 'India']",
                "[subject: 'Dev Gill', predicate: 'ocupación', object: 'modelo']",
                "[subject: 'India', predicate: 'subdividido en (división administrativa)', object: 'Chhattisgarh']"
            ],
            "El modelo Dev Gill es de nacionalidad india, que contiene la subdivisión Chhattisgarh"
        ),
        (
            [
                "[subject: 'Johann Reif', predicate: 'lugar de fallecimiento', object: 'Kritzendorf']",
                "[subject: 'Johann Reif', predicate: 'miembro del partido político', object: 'Partido Socialdemócrata de Austria']",
                "[subject: 'Kritzendorf', predicate: 'se encuentra en el huso horario', object: 'UTC+02:00']",
                "[subject: 'Partido Socialdemócrata de Austria', predicate: 'país', object: 'Austria']",
                "[subject: 'Kritzendorf', predicate: 'situado en la entidad territorial administrativa', object: 'Klosterneuburg']"
            ],
            "Johann Reif falleció en Kritzendorf y fue miembro del Partido Socialdemócrata de Austria. Kritzendorf se encuentra en el huso UTC+02:00 y está situado en la entidad territorial administrativa Klosterneuburg. El país del Partido Socialdemócrata de Austria es Austria"
        )
    ],
    "it": [
        (
            [
                "[subject: 'grotte di Altamira', predicate: 'unità amministrativa in cui è situato', object: 'Santillana del Mar']",
                "[subject: 'grotte di Altamira', predicate: 'continente', object: 'Europa']",
                "[subject: 'grotte di Altamira', predicate: 'pscoperto o inventato da', object: 'Marcelino Sanz de Sautuola']"
            ],
            "Le grotte di Altamira sono situate presso il comune di Santillana del Mar, in Europa e sono state scoperte da Marcelino Sanz de Sautuola"
        ),
        (
            [
                "[subject: 'Hansung University Design campus', predicate: 'paese', object: 'Corea del Sud']",
                "[subject: 'Hansung University Design campus', predicate: 'unità amministrativa in cui è situato', object: 'Seul']",
                "[subject: 'Corea del Sud', predicate: 'fuso orario', object: 'UTC+9']",
                "[subject: 'Corea del Sud', predicate: 'membro di', object: 'Organizzazione delle Nazioni Unite']"
            ],
            "La Hansung University Design campus si trova in Corea del Sud, a Seul. La Corea del Sud ha fuso orario UTC+9 ed è membro dell'Organizzazione delle Nazioni Unite"
        ),
        (
            [
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'paese', object: 'Austria']",
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'unità amministrativa in cui è situato', object: 'Vienna']",
                "[subject: 'Austria', predicate: 'organo esecutivo', object: 'Governo federale dell'Austria']",
                "[subject: 'Austria', predicate: 'situato sul mare / lago / fiume', object: 'Morava']",
                "[subject: 'Austria', predicate: 'organo legislativo', object: 'Assemblea Federale (Austria)']"
            ],
            "Il Monumento ai combattenti ebrei si trova in Austria, presso l'unità amministrativa di Vienna. L'Austria, attraverso cui scorre il fiume Morava,  ha come organo esecutivo il governo federale dell'austria e come organo legislativo l'assemblea Federale"
        )
    ]
}

NEGATIVE_ONE_SHOT_EXAMPLES = {
    "en":
        (
            [
                "[subject: 'Ligia Petit', predicate: 'place of birth', object: 'Maracay']",
                "[subject: 'Ligia Petit', predicate: 'occupation', object: 'model']",
                "[subject: 'Maracay', predicate: 'twinned administrative body', object: 'Salta']",
                "[subject: 'model', predicate: 'partially coincident with', object: 'fashion model']",
                "[subject: 'Maracay', predicate: 'located in the administrative territorial entity', object: 'Municipio Girardot']"
            ],
            "Liga Petit was born in Maracay, city in north-central Venezuela, near the Caribbean coast, and is the capital and most important city of the state of Aragua. She works as a fotomodel. Maracay is twinned with Malla and is located in Municipio Pacianot. The profession of a model partly coincides with that of a model."
        ),
    "es": 
        (
            [
                "[subject: 'Johann Reif', predicate: 'lugar de fallecimiento', object: 'Kritzendorf']",
                "[subject: 'Johann Reif', predicate: 'miembro del partido político', object: 'Partido Socialdemócrata de Austria']",
                "[subject: 'Kritzendorf', predicate: 'se encuentra en el huso horario', object: 'UTC+02:00']",
                "[subject: 'Partido Socialdemócrata de Austria', predicate: 'país', object: 'Austria']",
                "[subject: 'Kritzendorf', predicate: 'situado en la entidad territorial administrativa', object: 'Klosterneuburg']"
            ],
            " Johann Reif, político socialdemócrata austriaco, falleció en Kritzendorf y fue miembro del Partido Socialdemócrata de Austria. Kritzendorf se encuentra en el huso UTC+02:00 y está situado en la entidad territorial administrativa Klosterneuburg, una pequeña ciudad de Austria, en el Bundesland de Baja Austria. El país del Partido Socialdemócrata de Austria es Austria."
        ),
    "it":
        (
            [
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'paese', object: 'Austria']",
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'unità amministrativa in cui è situato', object: 'Vienna']",
                "[subject: 'Austria', predicate: 'organo esecutivo', object: 'Governo federale dell'Austria']",
                "[subject: 'Austria', predicate: 'situato sul mare / lago / fiume', object: 'Morava']",
                "[subject: 'Austria', predicate: 'organo legislativo', object: 'Assemblea Federale (Austria)']"
            ],
            "Monumento a los combatientes judíos (Viena) è Austria, presso l'unità amministrativa di Vienna, Austria. L'Austria, paese dell’europa centrale, attraverso cui scorre il fiume Morava, più importante fiume della Moravia, ha come organo esecutivo il governo dell'austria e come organo l'assemblea Federale"
        )
}



NEGATIVE_FEW_SHOT_EXAMPLES = {
    "en": [
        (
            [
                "[subject: 'Kewpie doll', predicate: 'discoverer or inventor', object: 'Rose O'Neill']",
                "[subject: 'Kewpie doll', predicate: 'creator', object: 'Rose O'Neill']",
                "[subject: 'Kewpie doll', predicate: 'penciller', object: 'Rose O'Neill']"
            ],
            "Kwepie doll, brand of dolls and figurines, is discovered, is created, and is drawn by Rose O'Neill"
        ),
        (
            [
                "[subject: 'Polsterberg Pumphouse', predicate: 'country', object: 'Germany']",
                "[subject: 'Polsterberg Pumphouse', predicate: 'located in the administrative territorial entity', object: 'Clausthal-Zellerfeld']",
                "[subject: 'Germany', predicate: 'head of state', object: 'Frank-Walter Steinmeier']",
                "[subject: 'Germany', predicate: 'shares border with', object: 'Belgium']"
            ],
            "The Polsterberg Pumphouse was in Germany and specifically in Clausthal-Zellerfeld. Germany borders with Belgium."
        ),
        (
            [
                "[subject: 'Ligia Petit', predicate: 'place of birth', object: 'Maracay']",
                "[subject: 'Ligia Petit', predicate: 'occupation', object: 'model']",
                "[subject: 'Maracay', predicate: 'twinned administrative body', object: 'Salta']",
                "[subject: 'model', predicate: 'partially coincident with', object: 'fashion model']",
                "[subject: 'Maracay', predicate: 'located in the administrative territorial entity', object: 'Municipio Girardot']"
            ],
            "Liga Petit was born in Maracay, city in north-central Venezuela, near the Caribbean coast, and is the capital and most important city of the state of Aragua. She works as a fotomodel. Maracay is twinned with Malla and is located in Municipio Pacianot. The profession of a model partly coincides with that of a model."
        )
    ],
    "es": [
        (
            [
                "[subject: 'Alois Pupp', predicate: 'país de nacionalidad', object: 'Italia']",
                "[subject: 'Alois Pupp', predicate: 'lugar de fallecimiento', object: 'Novoli']", 
                "[subject: 'Alois Pupp', predicate: 'cargo ocupado', object: 'Florencia']"
            ],
            "Alois Pupp, político italiano, sudtirolés de lengua ladina, es de nacionalidad italiana, falleció en Bresanona y ocupó el cargo de presidente del Tirol del Sur."
        ),
        (
            [
                "[subject: 'Dev Gill', predicate: 'país de nacionalidad', object: 'India']",
                "[subject: 'Dev Gill', predicate: 'ocupación', object: 'modelo']",
                "[subject: 'India', predicate: 'subdividido en (división administrativa)', object: 'Chhattisgarh']"
            ],
            "El modelo Devinder Singh Gill, actor y modelo indio que trabaja predominantemente en telug, es de nacionalidad india, que contiene la subdivisión Chhattisgarh."
        ),
        (
            [
                "[subject: 'Johann Reif', predicate: 'lugar de fallecimiento', object: 'Kritzendorf']",
                "[subject: 'Johann Reif', predicate: 'miembro del partido político', object: 'Partido Socialdemócrata de Austria']",
                "[subject: 'Kritzendorf', predicate: 'se encuentra en el huso horario', object: 'UTC+02:00']",
                "[subject: 'Partido Socialdemócrata de Austria', predicate: 'país', object: 'Austria']",
                "[subject: 'Kritzendorf', predicate: 'situado en la entidad territorial administrativa', object: 'Klosterneuburg']"
            ],
            " Johann Reif, político socialdemócrata austriaco, falleció en Kritzendorf y fue miembro del Partido Socialdemócrata de Austria. Kritzendorf se encuentra en el huso UTC+02:00 y está situado en la entidad territorial administrativa Klosterneuburg, una pequeña ciudad de Austria, en el Bundesland de Baja Austria. El país del Partido Socialdemócrata de Austria es Austria."
        )
    ],
    "it": [
        (
            [
                "[subject: 'grotte di Altamira', predicate: 'unità amministrativa in cui è situato', object: 'Santillana del Mar']",
                "[subject: 'grotte di Altamira', predicate: 'continente', object: 'Europa']",
                "[subject: 'grotte di Altamira', predicate: 'pscoperto o inventato da', object: 'Marcelino Sanz de Sautuola']"
            ],
            "La Grotta di Altamira, dove sono presenti pitture parietali paleolitiche, sono situate presso Santillana del mare, in Europa e sono scoperte da Marcello Sanzo de Sautuola"
        ),
        (
            [
                "[subject: 'Hansung University Design campus', predicate: 'paese', object: 'Corea del Sud']",
                "[subject: 'Hansung University Design campus', predicate: 'unità amministrativa in cui è situato', object: 'Seul']",
                "[subject: 'Corea del Sud', predicate: 'fuso orario', object: 'UTC+9']",
                "[subject: 'Corea del Sud', predicate: 'membro di', object: 'Organizzazione delle Nazioni Unite']"
            ],
            "L’università del Design si trova in Corea, nella capitale,  Seul. La Corea del Sud ha fuso orario UTC+8 ed è membro dell'ONU"
        ),
        (
            [
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'paese', object: 'Austria']",
                "[subject: 'Monumento a los combatientes judíos (Viena)', predicate: 'unità amministrativa in cui è situato', object: 'Vienna']",
                "[subject: 'Austria', predicate: 'organo esecutivo', object: 'Governo federale dell'Austria']",
                "[subject: 'Austria', predicate: 'situato sul mare / lago / fiume', object: 'Morava']",
                "[subject: 'Austria', predicate: 'organo legislativo', object: 'Assemblea Federale (Austria)']"
            ],
            "Monumento a los combatientes judíos (Viena) è Austria, presso l'unità amministrativa di Vienna, Austria. L'Austria, paese dell’europa centrale, attraverso cui scorre il fiume Morava, più importante fiume della Moravia, ha come organo esecutivo il governo dell'austria e come organo l'assemblea Federale"
        )
    ]
}
