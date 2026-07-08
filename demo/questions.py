"""Curated demo questions with knowledge graph subgraphs.

Each question includes:
- question: The natural language question
- answer: Expected answer
- q_entity: Topic entities in the question
- graph: KG subgraph as list of (subject, predicate, object) triples
"""

CURATED_QUESTIONS = [
    # --- People & Positions ---
    {
        "id": "obama_president",
        "question": "Who was the president of the United States born in Hawaii?",
        "answer": "Barack Obama",
        "q_entity": ["United States"],
        "graph": [
            ("Barack Obama", "president of", "United States"),
            ("Barack Obama", "born in", "Hawaii"),
            ("Barack Obama", "nationality", "United States"),
            ("Barack Obama", "educated at", "Harvard Law School"),
            ("Barack Obama", "mother", "Ann Dunham"),
            ("United States", "has capital", "Washington, D.C."),
            ("Hawaii", "part of", "United States"),
        ],
    },
    {
        "id": "einstein_university",
        "question": "Which university did Albert Einstein work at?",
        "answer": "Princeton University",
        "q_entity": ["Albert Einstein"],
        "graph": [
            ("Albert Einstein", "worked at", "Princeton University"),
            ("Albert Einstein", "field of work", "Physics"),
            ("Albert Einstein", "born in", "Ulm"),
            ("Albert Einstein", "nationality", "German"),
            ("Princeton University", "located in", "Princeton, New Jersey"),
            ("Princeton University", "type", "University"),
        ],
    },
    # --- Geography ---
    {
        "id": "nile_country",
        "question": "What country is the Nile River located in?",
        "answer": "Egypt",
        "q_entity": ["Nile River"],
        "graph": [
            ("Nile River", "located in", "Egypt"),
            ("Nile River", "length", "6,650 km"),
            ("Nile River", "flows through", "Sudan"),
            ("Nile River", "flows through", "Uganda"),
            ("Egypt", "capital", "Cairo"),
            ("Sudan", "capital", "Khartoum"),
        ],
    },
    {
        "id": "australia_capital",
        "question": "What is the capital of Australia?",
        "answer": "Canberra",
        "q_entity": ["Australia"],
        "graph": [
            ("Australia", "capital", "Canberra"),
            ("Canberra", "located in", "Australian Capital Territory"),
            ("Australia", "continent", "Australia"),
            ("Australia", "largest city", "Sydney"),
            ("Canberra", "founded", "1913"),
        ],
    },
    # --- Literature & Arts ---
    {
        "id": "orwell_1984",
        "question": "Who wrote the novel 1984?",
        "answer": "George Orwell",
        "q_entity": ["1984"],
        "graph": [
            ("1984", "author", "George Orwell"),
            ("1984", "genre", "Dystopian fiction"),
            ("1984", "first published", "1949"),
            ("George Orwell", "nationality", "British"),
            ("George Orwell", "real name", "Eric Arthur Blair"),
            ("George Orwell", "other work", "Animal Farm"),
        ],
    },
    {
        "id": "mona_lisa",
        "question": "Who painted the Mona Lisa?",
        "answer": "Leonardo da Vinci",
        "q_entity": ["Mona Lisa"],
        "graph": [
            ("Mona Lisa", "artist", "Leonardo da Vinci"),
            ("Mona Lisa", "location", "Louvre Museum"),
            ("Mona Lisa", "created", "1503"),
            ("Leonardo da Vinci", "nationality", "Italian"),
            ("Leonardo da Vinci", "field", "Painting"),
            ("Louvre Museum", "located in", "Paris"),
        ],
    },
    # --- Science ---
    {
        "id": "gold_symbol",
        "question": "What element has the chemical symbol Au?",
        "answer": "Gold",
        "q_entity": ["Au"],
        "graph": [
            ("Gold", "chemical symbol", "Au"),
            ("Gold", "atomic number", "79"),
            ("Gold", "category", "Transition metal"),
            ("Gold", "discovered by", "Ancient civilizations"),
            ("Au", "derived from", "Aurum"),
        ],
    },
    {
        "id": "relativity",
        "question": "Who proposed the theory of relativity?",
        "answer": "Albert Einstein",
        "q_entity": ["theory of relativity"],
        "graph": [
            ("Theory of relativity", "proposed by", "Albert Einstein"),
            ("Theory of relativity", "year", "1905"),
            ("Theory of relativity", "type", "Physics theory"),
            ("Albert Einstein", "field of work", "Physics"),
            ("Albert Einstein", "born in", "Ulm"),
            ("Albert Einstein", "employer", "Princeton University"),
        ],
    },
    # --- Sports ---
    {
        "id": "nba_2023",
        "question": "Which team won the 2023 NBA championship?",
        "answer": "Denver Nuggets",
        "q_entity": ["2023 NBA championship"],
        "graph": [
            ("2023 NBA championship", "winner", "Denver Nuggets"),
            ("2023 NBA championship", "runner up", "Miami Heat"),
            ("2023 NBA championship", "date", "2023"),
            ("Denver Nuggets", "located in", "Denver, Colorado"),
            ("Denver Nuggets", "conference", "Western Conference"),
        ],
    },
    {
        "id": "messi_ballon",
        "question": "Who has won the most Ballon d'Or awards?",
        "answer": "Lionel Messi",
        "q_entity": ["Ballon d'Or"],
        "graph": [
            ("Lionel Messi", "Ballon d'Or wins", "8"),
            ("Lionel Messi", "nationality", "Argentine"),
            ("Lionel Messi", "plays for", "Inter Miami"),
            ("Lionel Messi", "position", "Forward"),
            ("Ballon d'Or", "type", "Football award"),
        ],
    },
    # --- History ---
    {
        "id": "wwii_end",
        "question": "In what year did World War II end?",
        "answer": "1945",
        "q_entity": ["World War II"],
        "graph": [
            ("World War II", "end date", "1945"),
            ("World War II", "start date", "1939"),
            ("World War II", "result", "Allied victory"),
            ("World War II", "participants", "Allied Powers"),
            ("World War II", "participants", "Axis Powers"),
        ],
    },
    {
        "id": "moon_walk",
        "question": "Who was the first person to walk on the Moon?",
        "answer": "Neil Armstrong",
        "q_entity": ["Moon"],
        "graph": [
            ("Moon landing", "first person", "Neil Armstrong"),
            ("Moon landing", "mission", "Apollo 11"),
            ("Moon landing", "date", "1969"),
            ("Neil Armstrong", "nationality", "American"),
            ("Neil Armstrong", "occupation", "Astronaut"),
            ("Apollo 11", "launched by", "NASA"),
        ],
    },
]
