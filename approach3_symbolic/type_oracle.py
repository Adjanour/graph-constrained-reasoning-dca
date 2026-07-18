"""
type_oracle.py
==============
Purely symbolic type oracle using the Freebase ontology schema.

The Freebase schema (present in every WebQSP/CWQ subgraph) encodes:
  - Entity types via common.topic.notable_types
  - Property domains/ranges via rdf-schema#domain and rdf-schema#range
  - Property type constraints via type.property.expected_type

None of this requires embeddings. It is ground-truth structural
information from the KG itself.

Usage
-----
oracle = TypeOracle.from_graph(data["graph"])
answer_types = oracle.infer_answer_types(data["question"])
for each candidate path hop:
    oracle.is_admissible(relation, tail_entity, answer_types, hop, max_hop)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Human-readable Freebase type constants
# (Extracted from common.topic.notable_types in the WebQSP subgraphs)
# ---------------------------------------------------------------------------

_PERSON_TYPES: FrozenSet[str] = frozenset(
    {
        "Person",
        "Deceased Person",
        "Politician",
        "Musical Artist",
        "Author",
        "Academic",
        "Film director",
        "Film actor",
        "Inventor",
        "Military Person",
        "Religious Leader",
        "TV Actor",
        "Professional Athlete",
        "Olympic athlete",
        "Singer",
        "Composer",
    }
)

_LOCATION_TYPES: FrozenSet[str] = frozenset(
    {
        "Location",
        "Country",
        "City/Town/Village",
        "US State",
        "Administrative Division",
        "US County",
        "Place of interment",
    }
)

_ORGANIZATION_TYPES: FrozenSet[str] = frozenset(
    {
        "Organization",
        "Government Agency",
        "Educational Institution",
        "College/University",
        "School",
        "Sports Team",
        "Sports Facility",
        "Company",
        "Venue",
    }
)

_CREATIVE_WORK_TYPES: FrozenSet[str] = frozenset(
    {
        "Film",
        "Book",
        "Musical Recording",
        "Musical Album",
        "TV Program",
        "TV Series",
        "TV Season",
        "Written Work",
        "Composition",
        "Play",
    }
)

_DATE_TYPES: FrozenSet[str] = frozenset(
    {
        "Date/Time",
    }
)

_LANGUAGE_TYPES: FrozenSet[str] = frozenset(
    {
        "Human Language",
    }
)

_AWARD_TYPES: FrozenSet[str] = frozenset(
    {
        "Award",
        "Award honor",
    }
)

_PROFESSION_TYPES: FrozenSet[str] = frozenset(
    {
        "Profession",
    }
)

_GENRE_TYPES: FrozenSet[str] = frozenset(
    {
        "TV Genre",
        "Music Genre",
        "Musical genre",
        "Film genre",
        "Literary Genre",
        "Media Genre",
    }
)


# ---------------------------------------------------------------------------
# Answer type inference from question words
# ---------------------------------------------------------------------------

ANSWER_TYPE_MAP: Dict[str, FrozenSet[str]] = {
    "nationality": _PERSON_TYPES | _LOCATION_TYPES,
    "country": _LOCATION_TYPES,
    "city": _LOCATION_TYPES,
    "location": _LOCATION_TYPES,
    "director": _PERSON_TYPES,
    "actor": _PERSON_TYPES,
    "person": _PERSON_TYPES,
    "who": _PERSON_TYPES,
    "film": _CREATIVE_WORK_TYPES,
    "movie": _CREATIVE_WORK_TYPES,
    "date": _DATE_TYPES,
    "when": _DATE_TYPES,
    "year": _DATE_TYPES,
    "organization": _ORGANIZATION_TYPES,
    "company": _ORGANIZATION_TYPES,
    "language": _LANGUAGE_TYPES,
    "award": _AWARD_TYPES,
    "prize": _AWARD_TYPES,
    "genre": _GENRE_TYPES,
    "sport": frozenset({"Sports Team", "Sport"}),
    "team": frozenset({"Sports Team"}),
    "profession": _PROFESSION_TYPES,
    "job": _PROFESSION_TYPES,
    "occupation": _PROFESSION_TYPES,
}

_QUESTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnationality\b|\bcitizenship\b", re.I), "nationality"),
    (re.compile(r"\bcountry\b|\bnation\b", re.I), "country"),
    (re.compile(r"\bcity\b|\btown\b|\bvillage\b", re.I), "city"),
    (re.compile(r"\blocation\b|\bwhere\b|\bplace\b", re.I), "location"),
    (re.compile(r"\bdirector\b|\bdirected by\b", re.I), "director"),
    (re.compile(r"\bactor\b|\bstarred\b|\bplayed by\b", re.I), "actor"),
    (re.compile(r"\bwho\b", re.I), "who"),
    (re.compile(r"\bfilm\b|\bmovie\b", re.I), "film"),
    (re.compile(r"\bwhen\b|\bdate\b|\byear\b", re.I), "date"),
    (re.compile(r"\baward\b|\bpriz\b|\bhonor\b", re.I), "award"),
    (re.compile(r"\blanguage\b|\bspoken\b|\bspeak\b", re.I), "language"),
    (re.compile(r"\bgenre\b|\bstyle\b", re.I), "genre"),
    (re.compile(r"\bteam\b|\bclub\b", re.I), "team"),
    (re.compile(r"\bsport\b", re.I), "sport"),
    (re.compile(r"\bcompany\b|\bbusiness\b|\bcorporation\b", re.I), "company"),
    (re.compile(r"\borganization\b|\binstitution\b", re.I), "organization"),
    (re.compile(r"\bprofession\b|\bjob\b|\boccupation\b", re.I), "profession"),
    (re.compile(r"\bwhat.*profession|what.*do\b", re.I), "profession"),
]


# ---------------------------------------------------------------------------
# Relation schema: domain and range constraints
# ---------------------------------------------------------------------------
# Key: Freebase relation ID (e.g., "people.person.place_of_birth")
# Values: human-readable type names that appear in the subgraph

_RELATION_SCHEMA: Dict[str, Dict[str, FrozenSet[str]]] = {
    # People
    "people.person.nationality": {
        "domain": _PERSON_TYPES,
        "range": _LOCATION_TYPES,
    },
    "people.person.place_of_birth": {
        "domain": _PERSON_TYPES,
        "range": _LOCATION_TYPES,
    },
    "people.person.profession": {
        "domain": _PERSON_TYPES,
        "range": _PROFESSION_TYPES,
    },
    "people.person.gender": {
        "domain": _PERSON_TYPES,
        "range": frozenset({"Gender"}),
    },
    "people.person.children": {
        "domain": _PERSON_TYPES,
        "range": _PERSON_TYPES,
    },
    "people.person.parents": {
        "domain": _PERSON_TYPES,
        "range": _PERSON_TYPES,
    },
    "people.person.spouse_s": {
        "domain": _PERSON_TYPES,
        "range": _PERSON_TYPES,
    },
    "people.deceased_person.place_of_death": {
        "domain": frozenset({"Deceased Person"}),
        "range": _LOCATION_TYPES,
    },
    "people.person.education": {
        "domain": _PERSON_TYPES,
        "range": frozenset({"Education"}),
    },
    "people.person.languages": {
        "domain": _PERSON_TYPES,
        "range": _LANGUAGE_TYPES,
    },
    "people.ethnicity.people": {
        "domain": frozenset({"Ethnicity"}),
        "range": _PERSON_TYPES,
    },
    # Film / TV
    "film.film.directed_by": {
        "domain": _CREATIVE_WORK_TYPES,
        "range": _PERSON_TYPES,
    },
    "film.film.starring": {
        "domain": _CREATIVE_WORK_TYPES,
        "range": _PERSON_TYPES,
    },
    "film.performance.actor": {
        "domain": frozenset({"Film performance"}),
        "range": _PERSON_TYPES,
    },
    "film.film.country": {
        "domain": _CREATIVE_WORK_TYPES,
        "range": _LOCATION_TYPES,
    },
    "film.film.language": {
        "domain": _CREATIVE_WORK_TYPES,
        "range": _LANGUAGE_TYPES,
    },
    "film.film.genre": {
        "domain": _CREATIVE_WORK_TYPES,
        "range": _GENRE_TYPES,
    },
    # Broadcast/TV
    "tv.tv_program.country_of_origin": {
        "domain": frozenset({"TV Program"}),
        "range": _LOCATION_TYPES,
    },
    "tv.tv_program.genre": {
        "domain": frozenset({"TV Program"}),
        "range": _GENRE_TYPES,
    },
    "tv.tv_program.languages": {
        "domain": frozenset({"TV Program"}),
        "range": _LANGUAGE_TYPES,
    },
    # Location
    "location.location.containedby": {
        "domain": _LOCATION_TYPES,
        "range": _LOCATION_TYPES,
    },
    "location.location.contains": {
        "domain": _LOCATION_TYPES,
        "range": _LOCATION_TYPES,
    },
    "location.country.capital": {
        "domain": _LOCATION_TYPES,
        "range": _LOCATION_TYPES,
    },
    "location.country.languages_spoken": {
        "domain": _LOCATION_TYPES,
        "range": _LANGUAGE_TYPES,
    },
    "location.country.administrative_divisions": {
        "domain": _LOCATION_TYPES,
        "range": _LOCATION_TYPES,
    },
    "location.country.form_of_government": {
        "domain": _LOCATION_TYPES,
        "range": frozenset({"Form of government"}),
    },
    "location.country.currency_used": {
        "domain": _LOCATION_TYPES,
        "range": frozenset({"Currency"}),
    },
    # Organization
    "organization.organization.headquarters": {
        "domain": _ORGANIZATION_TYPES,
        "range": _LOCATION_TYPES,
    },
    "organization.organization.founders": {
        "domain": _ORGANIZATION_TYPES,
        "range": _PERSON_TYPES,
    },
    "organization.organization.place_founded": {
        "domain": _ORGANIZATION_TYPES,
        "range": _LOCATION_TYPES,
    },
    # Government
    "government.politician.government_positions_held": {
        "domain": _PERSON_TYPES,
        "range": frozenset({"Government Position Held"}),
    },
    "government.politician.party": {
        "domain": _PERSON_TYPES,
        "range": frozenset({"Political Party"}),
    },
    # Sports
    "sports.pro_athlete.teams": {
        "domain": frozenset({"Professional Athlete"}),
        "range": frozenset({"Sports Team", "Sports Team Roster"}),
    },
    "sports.sports_team.sport": {
        "domain": frozenset({"Sports Team"}),
        "range": frozenset({"Sport"}),
    },
    # Music
    "music.artist.genre": {
        "domain": frozenset({"Musical Artist"}),
        "range": _GENRE_TYPES,
    },
    "music.composition.composer": {
        "domain": frozenset({"Composition"}),
        "range": _PERSON_TYPES,
    },
    # Award
    "award.award_honor.award_winner": {
        "domain": _AWARD_TYPES,
        "range": _PERSON_TYPES,
    },
    "award.award_honor.honored_for": {
        "domain": _AWARD_TYPES,
        "range": _CREATIVE_WORK_TYPES,
    },
    # Book
    "book.written_work.author": {
        "domain": frozenset({"Written Work", "Book"}),
        "range": _PERSON_TYPES,
    },
    "book.written_work.original_language": {
        "domain": frozenset({"Written Work", "Book"}),
        "range": _LANGUAGE_TYPES,
    },
}


# ---------------------------------------------------------------------------
# Schema predicate detection (for auto-mining)
# ---------------------------------------------------------------------------
# These predicates are Freebase schema/metadata, not data relations that
# carry entity-to-entity connectivity.  We skip them when mining relation
# domain/range from the graph.

_SCHEMA_PREDICATES: FrozenSet[str] = frozenset({
    "common.topic.notable_types",
    "freebase.type_hints.included_types",
    "freebase.type_profile.strict_included_types",
    "rdf-schema#domain",
    "rdf-schema#range",
    "type.property.expected_type",
    "type.property.schema",
    "type.type.properties",
    "type.type.expected_by",
    "type.type.domain",
    "freebase.type_profile.published",
    "freebase.type_profile.kind",
    "freebase.type_profile.equivalent_topic",
    "freebase.equivalent_topic.equivalent_type",
    "freebase.type_kind.types",
    "freebase.type_profile.tasks",
    "freebase.valuenotation.is_reviewed",
    "freebase.valuenotation.has_value",
})


def _is_schema_predicate(r: str) -> bool:
    """Return True if *r* is a Freebase schema/metadata predicate (not a data relation)."""
    if r in _SCHEMA_PREDICATES:
        return True
    if r.startswith("type."):
        return True
    if r.startswith("base.") and "schema" in r:
        return True
    return False


# ---------------------------------------------------------------------------
# TypeOracle
# ---------------------------------------------------------------------------


class TypeOracle:
    """
    Purely symbolic type oracle using the Freebase ontology schema.

    Two sources of relation domain/range information:
      1. Hand-curated ``_RELATION_SCHEMA`` (38 relations, high precision)
      2. Auto-mined from actual data triples in the subgraph (full coverage)

    When ``from_graph()`` is called, it automatically mines relation schema
    from the subgraph data, so ALL relations in the subgraph have at least
    some domain/range information.

    Usage
    -----
    oracle = TypeOracle.from_graph(data["graph"])
    oracle.is_admissible(relation, tail_entity, answer_types, hop, max_hop)
    """

    def __init__(
        self,
        entity_type_map: Dict[str, FrozenSet[str]] | None = None,
        relation_schema: Dict[str, Dict[str, FrozenSet[str]]] | None = None,
        mined_schema: Dict[str, Dict[str, FrozenSet[str]]] | None = None,
    ):
        self._entity_types: Dict[str, FrozenSet[str]] = entity_type_map or {}
        self._schema: Dict[str, Dict[str, FrozenSet[str]]] = relation_schema or dict(
            _RELATION_SCHEMA
        )
        self._mined_schema: Dict[str, Dict[str, FrozenSet[str]]] = mined_schema or {}

    # ------------------------------------------------------------------
    # Construction from raw graph
    # ------------------------------------------------------------------

    @classmethod
    def from_graph(cls, graph_triples: List[List[str]]) -> "TypeOracle":
        """
        Build a TypeOracle from a Freebase subgraph.

        Extracts:
          - Entity types from ``common.topic.notable_types`` triples
          - Entity type aliases from ``freebase.type_hints.included_types``
          - **Relation domain/range** from actual data triples via type
            aggregation (NEW — covers ALL relations, not just 38 hand-curated)

        Parameters
        ----------
        graph_triples : list of [head, relation, tail]
            The raw graph from dataset["graph"]
        """
        entity_types: Dict[str, Set[str]] = defaultdict(set)

        for h, r, t in graph_triples:
            if r == "common.topic.notable_types":
                entity_types[h].add(t)
            elif r == "freebase.type_hints.included_types":
                # "Topic" is a supertype of everything — skip
                if t != "Topic":
                    entity_types[h].add(t)
            elif r == "freebase.type_profile.strict_included_types":
                if t != "Topic":
                    entity_types[h].add(t)

        # Freeze
        frozen: Dict[str, FrozenSet[str]] = {
            e: frozenset(ts) for e, ts in entity_types.items()
        }

        # NEW: mine relation schema from data triples
        mined = cls._mine_relation_schema(graph_triples, frozen)

        return cls(entity_type_map=frozen, mined_schema=mined)

    # ------------------------------------------------------------------
    # Auto-mine relation schema from data triples
    # ------------------------------------------------------------------

    @staticmethod
    def _mine_relation_schema(
        graph_triples: List[List[str]],
        entity_type_map: Dict[str, FrozenSet[str]],
    ) -> Dict[str, Dict[str, FrozenSet[str]]]:
        """
        Infer relation domain/range from actual data triples in the subgraph.

        For each non-schema triple ``(h, r, t)``, looks up the entity types
        of ``h`` (→ domain) and ``t`` (→ range) and aggregates them per
        relation.  The result is a schema dict in the same format as
        ``_RELATION_SCHEMA``.

        This covers **all** relations in the subgraph, not just 38 hand-curated
        ones.  The mined ranges reflect actual usage patterns rather than
        Freebase's theoretical type hierarchy, so they tend to be *more*
        specific and more useful for filtering.
        """
        mined: Dict[str, Dict[str, set]] = defaultdict(lambda: {"domain": set(), "range": set()})

        for h, r, t in graph_triples:
            if _is_schema_predicate(r):
                continue
            h_types = entity_type_map.get(h, frozenset())
            t_types = entity_type_map.get(t, frozenset())
            if h_types:
                mined[r]["domain"].update(h_types)
            if t_types:
                mined[r]["range"].update(t_types)

        return {
            r: {"domain": frozenset(d["domain"]), "range": frozenset(d["range"])}
            for r, d in mined.items()
            if d["domain"] or d["range"]
        }

    # ------------------------------------------------------------------
    # Entity type access
    # ------------------------------------------------------------------

    def get_types(self, entity_name: str) -> FrozenSet[str]:
        """Return the known Freebase types for an entity."""
        return self._entity_types.get(entity_name, frozenset())

    # ------------------------------------------------------------------
    # Answer type inference from question (regex) + KG fallback
    # ------------------------------------------------------------------

    def infer_answer_types(self, question: str) -> FrozenSet[str]:
        """
        Infer expected answer types from the question string.

        Returns frozenset of human-readable type strings.
        Returns empty set (unconstrained) if no pattern matches.
        """
        q = self._normalize_question(question)
        matched: Set[str] = set()

        for pattern, type_key in _QUESTION_PATTERNS:
            if pattern.search(q):
                matched.update(ANSWER_TYPE_MAP.get(type_key, frozenset()))

        return frozenset(matched)

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Mask quoted entity mentions to avoid matching on them."""
        q = re.sub(r'"[^"]+"', " [ent] ", question)
        q = re.sub(r"'[^']+'", " [ent] ", q)
        return q

    # ------------------------------------------------------------------
    # Answer type inference from KG (no regex, no NLU)
    # ------------------------------------------------------------------

    def infer_answer_types_from_paths(
        self,
        paths: List[List[Tuple[str, str, str]]],
    ) -> FrozenSet[str]:
        """
        Infer answer types purely from the KG — no regex, no NLU.

        Uses the entity types of all terminal entities in the candidate
        *paths*.  This naturally reflects what types of answers are
        reachable from the topic entities via the KG schema.

        This is used as a fallback when ``infer_answer_types()`` returns
        empty.  Because it relies on actual KG entities (not question
        wording), it's robust to unusual question phrasing.

        Parameters
        ----------
        paths : list of list of (head, relation, tail) tuples
            Candidate KG paths (e.g., from DFS).

        Returns
        -------
        frozenset of human-readable type strings.
        """
        answer_types: Set[str] = set()
        for path in paths:
            terminal = path[-1][2]
            etypes = self._entity_types.get(terminal, frozenset())
            answer_types.update(etypes)
        return frozenset(answer_types)

    # ------------------------------------------------------------------
    # Gate 1: Answer type gate (terminal hop only)
    # ------------------------------------------------------------------

    def type_gate(
        self,
        entity_name: str,
        answer_types: FrozenSet[str],
        hop: int,
        max_hop: int,
    ) -> bool:
        """
        Check entity type compatibility at the terminal hop.

        At intermediate hops: always allow.
        At terminal hop: allow if entity's types intersect answer_types.
        If entity has no recorded types: allow (conservative).
        If answer_types is empty (no pattern matched): allow.
        """
        if hop < max_hop:
            return True
        if not answer_types:
            return True

        etypes = self._entity_types.get(entity_name, frozenset())
        if not etypes:
            return True

        return bool(etypes & answer_types)

    # ------------------------------------------------------------------
    # Gate 2: Property range gate
    # ------------------------------------------------------------------

    def range_gate(self, relation: str, tail_entity: str) -> bool:
        """
        Check whether tail_entity's type is compatible with the
        relation's declared range.

        Checks two sources in order:
          1. Hand-curated ``_RELATION_SCHEMA`` (38 relations, authoritative)
          2. Auto-mined schema from data triples (all relations, data-driven)

        Returns True if:
          - relation not in any known schema (conservative)
          - tail entity has no recorded types (conservative)
          - tail entity types intersect the relation's range
        """
        rel_schema = self._mined_schema.get(relation)
        if rel_schema is None:
            rel_schema = self._schema.get(relation)
        if rel_schema is None:
            return True

        range_types = rel_schema.get("range", frozenset())
        if not range_types:
            return True

        etypes = self._entity_types.get(tail_entity, frozenset())
        if not etypes:
            return True

        return bool(etypes & range_types)

    # ------------------------------------------------------------------
    # Combined admission check
    # ------------------------------------------------------------------

    def is_admissible(
        self,
        relation: str,
        tail_entity: str,
        answer_types: FrozenSet[str],
        hop: int,
        max_hop: int,
    ) -> bool:
        """
        Full symbolic admission check for a candidate edge.

        A candidate (relation, tail_entity) is admitted iff it
        passes ALL active gates.

        This replaces the product cosine score entirely.
        No embeddings, no forward passes, no threshold tuning.
        """
        # Gate 1: answer type constraint at terminal hop
        if not self.type_gate(tail_entity, answer_types, hop, max_hop):
            return False

        # Gate 2: property range constraint
        if not self.range_gate(relation, tail_entity):
            return False

        return True


# ---------------------------------------------------------------------------
# SIR type component helper
# ---------------------------------------------------------------------------


def compute_type_irrelevance(
    paths: List[tuple],
    oracle: TypeOracle,
    answer_types: FrozenSet[str],
    max_hop: int,
) -> float:
    """
    Compute SIR*_type for a set of candidate paths.

    SIR*_type = fraction of paths whose terminal entity type
    is incompatible with the inferred answer types.
    """
    if not paths or not answer_types:
        return 0.0

    n_irrelevant = 0
    for path in paths:
        terminal = path[-1]
        terminal_name = (
            terminal if isinstance(terminal, str) else terminal.get("id", "")
        )
        if not oracle.type_gate(terminal_name, answer_types, max_hop, max_hop):
            n_irrelevant += 1

    return n_irrelevant / len(paths)
