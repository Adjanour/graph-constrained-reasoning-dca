"""
type_oracle.py
==============
Purely symbolic type oracle using the Freebase ontology schema.

The Freebase schema encodes:
  - Domain → Type → Property hierarchy  (e.g. /people/person)
  - Expected value types on each property (RDFS range)
  - Expected domain types on each property (RDFS domain)
  - Relation composition patterns

None of this requires embeddings. It is ground-truth structural
information from the KG itself.

Integration point in GCR pipeline
----------------------------------
Called before BuildTrie (v1) and before each expansion step (v2).
Receives entity MIDs and relation strings directly from the KG
subgraph returned by GCR's entity-linking + BFS stages.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Freebase type system constants
# ---------------------------------------------------------------------------

# Top-level Freebase domains that indicate entity categories
# Used for coarse answer-type inference from questions
_NATIONALITY_TYPES: FrozenSet[str] = frozenset({
    "/location/country",
    "/location/nationality",
    "/people/nationality",
})

_PERSON_TYPES: FrozenSet[str] = frozenset({
    "/people/person",
    "/film/director",
    "/film/actor",
    "/music/artist",
    "/sports/pro_athlete",
    "/government/politician",
})

_FILM_TYPES: FrozenSet[str] = frozenset({
    "/film/film",
    "/film/tv_program",
})

_LOCATION_TYPES: FrozenSet[str] = frozenset({
    "/location/location",
    "/location/city",
    "/location/country",
    "/location/us_state",
    "/location/administrative_division",
})

_DATE_TYPES: FrozenSet[str] = frozenset({
    "/type/datetime",
})

_ORGANIZATION_TYPES: FrozenSet[str] = frozenset({
    "/organization/organization",
    "/business/company",
    "/education/educational_institution",
    "/government/government_agency",
})

# Mapping from question-level answer intent signals to compatible Freebase types
# These are the terminal-hop type constraints (ρ_e in the paper)
ANSWER_TYPE_MAP: Dict[str, FrozenSet[str]] = {
    "nationality":      _NATIONALITY_TYPES | _LOCATION_TYPES,
    "country":          _NATIONALITY_TYPES | _LOCATION_TYPES,
    "location":         _LOCATION_TYPES,
    "city":             _LOCATION_TYPES,
    "director":         _PERSON_TYPES,
    "actor":            _PERSON_TYPES,
    "person":           _PERSON_TYPES,
    "who":              _PERSON_TYPES | _ORGANIZATION_TYPES,
    "film":             _FILM_TYPES,
    "movie":            _FILM_TYPES,
    "show":             _FILM_TYPES,
    "date":             _DATE_TYPES,
    "when":             _DATE_TYPES,
    "year":             _DATE_TYPES,
    "organization":     _ORGANIZATION_TYPES,
    "company":          _ORGANIZATION_TYPES,
    "language":         frozenset({"/language/human_language"}),
    "award":            frozenset({"/award/award_honor", "/award/award"}),
    "genre":            frozenset({"/film/film_genre", "/music/genre"}),
    "sport":            frozenset({"/sports/sport"}),
    "team":             frozenset({"/sports/sports_team"}),
}

# Question-word patterns → answer type keys
# These are matched against the *entity-masked* question string
_QUESTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnationality\b|\bcitizenship\b",          re.I), "nationality"),
    (re.compile(r"\bcountry\b|\bnation\b",                    re.I), "country"),
    (re.compile(r"\bcity\b|\btown\b|\bvillage\b",             re.I), "city"),
    (re.compile(r"\blocation\b|\bwhere\b|\bplace\b",          re.I), "location"),
    (re.compile(r"\bdirector\b|\bdirected by\b",              re.I), "director"),
    (re.compile(r"\bactor\b|\bstarred\b|\bplayed by\b",       re.I), "actor"),
    (re.compile(r"\bwho\b",                                   re.I), "who"),
    (re.compile(r"\bfilm\b|\bmovie\b",                        re.I), "film"),
    (re.compile(r"\bshow\b|\bseries\b|\bprogram\b",           re.I), "show"),
    (re.compile(r"\bwhen\b|\bdate\b|\byear\b",                re.I), "date"),
    (re.compile(r"\baward\b|\bpriz\b|\bhonor\b",              re.I), "award"),
    (re.compile(r"\blanguage\b|\bspoken\b",                   re.I), "language"),
    (re.compile(r"\bgenre\b|\bstyle\b",                       re.I), "genre"),
    (re.compile(r"\bteam\b|\bclub\b",                         re.I), "team"),
    (re.compile(r"\bsport\b",                                 re.I), "sport"),
    (re.compile(r"\bcompany\b|\bbusiness\b|\bcorporation\b",  re.I), "company"),
    (re.compile(r"\borganization\b|\binstitution\b",          re.I), "organization"),
]


# ---------------------------------------------------------------------------
# Relation schema: domain and range constraints
# ---------------------------------------------------------------------------
# For each Freebase relation we record:
#   domain: set of types the subject (head) entity must have
#   range:  set of types the object (tail) entity must have
#
# These come directly from the Freebase RDF schema and are
# the RDFS domain/range triples in the ontology.
# Only a representative subset is hard-coded here; a full version
# loads from the Freebase schema dump at init time.

_RELATION_SCHEMA: Dict[str, Dict[str, FrozenSet[str]]] = {
    "film.film.directed_by": {
        "domain": _FILM_TYPES,
        "range":  _PERSON_TYPES,
    },
    "film.film.starring": {
        "domain": _FILM_TYPES,
        "range":  _PERSON_TYPES,
    },
    "film.performance.actor": {
        "domain": frozenset({"/film/performance"}),
        "range":  _PERSON_TYPES,
    },
    "film.film.country": {
        "domain": _FILM_TYPES,
        "range":  _LOCATION_TYPES,
    },
    "film.film.genre": {
        "domain": _FILM_TYPES,
        "range":  frozenset({"/film/film_genre"}),
    },
    "film.film.language": {
        "domain": _FILM_TYPES,
        "range":  frozenset({"/language/human_language"}),
    },
    "people.person.nationality": {
        "domain": _PERSON_TYPES,
        "range":  _NATIONALITY_TYPES,
    },
    "people.person.place_of_birth": {
        "domain": _PERSON_TYPES,
        "range":  _LOCATION_TYPES,
    },
    "people.person.profession": {
        "domain": _PERSON_TYPES,
        "range":  frozenset({"/people/profession"}),
    },
    "location.location.containedby": {
        "domain": _LOCATION_TYPES,
        "range":  _LOCATION_TYPES,
    },
    "organization.organization.headquarters": {
        "domain": _ORGANIZATION_TYPES,
        "range":  _LOCATION_TYPES,
    },
    "government.politician.party": {
        "domain": frozenset({"/government/politician"}),
        "range":  frozenset({"/government/political_party"}),
    },
    "music.artist.genre": {
        "domain": frozenset({"/music/artist"}),
        "range":  frozenset({"/music/genre"}),
    },
    "sports.pro_athlete.teams": {
        "domain": frozenset({"/sports/pro_athlete"}),
        "range":  frozenset({"/sports/sports_team_roster"}),
    },
    "award.award_honor.award_winner": {
        "domain": frozenset({"/award/award_honor"}),
        "range":  _PERSON_TYPES | _ORGANIZATION_TYPES,
    },
    "award.award_honor.honored_for": {
        "domain": frozenset({"/award/award_honor"}),
        "range":  _FILM_TYPES | frozenset({"/music/album", "/book/book"}),
    },
}

# Valid 2-hop relation composition patterns
# (r1, r2) → True means this sequence is semantically coherent
# Derived from Freebase CVT structure and common reasoning chains
_VALID_COMPOSITIONS: FrozenSet[Tuple[str, str]] = frozenset({
    # Film → director → nationality
    ("film.film.directed_by",       "people.person.nationality"),
    # Film → starring → actor → nationality
    ("film.film.starring",          "people.person.nationality"),
    ("film.film.starring",          "people.person.place_of_birth"),
    # Person → award → film
    ("award.award_honor.award_winner", "award.award_honor.honored_for"),
    # Location → containedby chains
    ("location.location.containedby", "location.location.containedby"),
    # Organization → headquarters → country
    ("organization.organization.headquarters", "location.location.containedby"),
    # Athlete → team roster → team
    ("sports.pro_athlete.teams",    "sports.sports_team_roster.team"),
    # Film → country of origin
    ("film.film.country",           "location.location.containedby"),
})


# ---------------------------------------------------------------------------
# TypeOracle class
# ---------------------------------------------------------------------------

class TypeOracle:
    """
    Purely symbolic type oracle using Freebase ontology.

    Three constraint levels (in order of application):
      1. Answer type gate      — terminal entity type vs question intent
      2. Property range gate   — tail entity type vs relation RDFS range
      3. Composition gate      — (r1, r2) pair vs valid composition patterns

    All operations are O(1) set lookups. No forward passes, no embeddings.

    Parameters
    ----------
    schema_path : str | Path, optional
        Path to a JSON file containing extended relation schema.
        If provided, merges with the built-in _RELATION_SCHEMA.
        Format: {"relation": {"domain": [...], "range": [...]}, ...}
    """

    def __init__(self, schema_path: Optional[str | Path] = None):
        self._schema: Dict[str, Dict[str, FrozenSet[str]]] = dict(_RELATION_SCHEMA)

        if schema_path is not None:
            self._load_schema(schema_path)

        # Cache: entity MID → set of Freebase types
        # Populated lazily from the KG subgraph
        self._entity_type_cache: Dict[str, FrozenSet[str]] = {}

    def _load_schema(self, path: str | Path) -> None:
        """Merge an external schema JSON into the built-in schema."""
        with open(path) as f:
            external = json.load(f)
        for rel, constraints in external.items():
            self._schema[rel] = {
                "domain": frozenset(constraints.get("domain", [])),
                "range":  frozenset(constraints.get("range",  [])),
            }

    # ------------------------------------------------------------------
    # Entity type registration
    # ------------------------------------------------------------------

    def register_entity_types(
        self,
        entity_type_map: Dict[str, List[str]]
    ) -> None:
        """
        Bulk-register entity → type mappings from the KG subgraph.

        Parameters
        ----------
        entity_type_map : dict
            {entity_mid: [freebase_type_string, ...]}
            e.g. {"m.01_xyz": ["/film/film", "/media_common/creative_work"]}
        """
        for eid, types in entity_type_map.items():
            self._entity_type_cache[eid] = frozenset(types)

    def get_entity_types(self, entity_mid: str) -> FrozenSet[str]:
        """Return the registered types for an entity, or empty set."""
        return self._entity_type_cache.get(entity_mid, frozenset())

    # ------------------------------------------------------------------
    # Answer type inference from question
    # ------------------------------------------------------------------

    def infer_answer_types(self, question: str) -> FrozenSet[str]:
        """
        Infer the expected Freebase types of the answer entity
        from the natural language question.

        Uses pattern matching on the entity-masked question.
        Returns a frozenset of compatible Freebase type strings.
        Returns frozenset() (unconstrained) if no pattern matches.

        Parameters
        ----------
        question : str
            Raw question string. Entity mentions are masked internally.
        """
        q_masked = self._mask_entities(question)
        matched_types: Set[str] = set()

        for pattern, type_key in _QUESTION_PATTERNS:
            if pattern.search(q_masked):
                matched_types.update(ANSWER_TYPE_MAP.get(type_key, frozenset()))

        return frozenset(matched_types)

    @staticmethod
    def _mask_entities(question: str) -> str:
        """
        Heuristically mask entity mentions from question string.

        We strip tokens that look like proper nouns (consecutive
        capitalized words) to isolate the relational intent.
        This is a heuristic — entity linking output can be used
        instead for more precise masking.
        """
        # Remove text in quotes (often entity names)
        masked = re.sub(r'"[^"]+"', '[ENT]', question)
        masked = re.sub(r"'[^']+'", '[ENT]', masked)
        # Mask runs of title-cased words (proper nouns)
        masked = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', '[ENT]', masked)
        return masked

    # ------------------------------------------------------------------
    # Gate 1: Answer type gate (terminal hop)
    # ------------------------------------------------------------------

    def type_gate(
        self,
        entity_mid: str,
        answer_types: FrozenSet[str],
        hop: int,
        max_hop: int,
    ) -> bool:
        """
        Check whether entity_mid is type-compatible with the
        expected answer types at the terminal hop.

        At intermediate hops, always returns True (no constraint).
        At the terminal hop, returns True iff at least one of the
        entity's registered types intersects with answer_types.

        If answer_types is empty (no pattern matched), returns True
        unconditionally — this is the safe fallback.

        Parameters
        ----------
        entity_mid   : str   Entity MID from Freebase
        answer_types : frozenset  Compatible types from infer_answer_types()
        hop          : int   Current hop depth (1-indexed)
        max_hop      : int   Maximum hop depth (L)
        """
        # Only apply at terminal hop
        if hop < max_hop:
            return True
        # No constraint inferred → unconstrained
        if not answer_types:
            return True

        entity_types = self.get_entity_types(entity_mid)
        # No type info registered → allow (conservative: don't prune unknowns)
        if not entity_types:
            return True

        return bool(entity_types & answer_types)

    # ------------------------------------------------------------------
    # Gate 2: Property range gate
    # ------------------------------------------------------------------

    def range_gate(
        self,
        relation: str,
        tail_entity_mid: str,
    ) -> bool:
        """
        Check whether tail_entity_mid has a type compatible with
        the RDFS range of relation.

        Returns True if:
          - relation is not in schema (unknown relation → unconstrained)
          - tail entity has no registered types (conservative: allow)
          - tail entity types intersect with relation's declared range

        Parameters
        ----------
        relation       : str  Freebase relation string
        tail_entity_mid: str  MID of the tail entity
        """
        if relation not in self._schema:
            return True

        range_types = self._schema[relation].get("range", frozenset())
        if not range_types:
            return True

        entity_types = self.get_entity_types(tail_entity_mid)
        if not entity_types:
            return True  # conservative

        return bool(entity_types & range_types)

    # ------------------------------------------------------------------
    # Gate 3: Relation composition gate
    # ------------------------------------------------------------------

    def composition_gate(
        self,
        prev_relation: Optional[str],
        curr_relation: str,
    ) -> bool:
        """
        Check whether the transition prev_relation → curr_relation
        is a valid composition pattern.

        At hop 1, prev_relation is None → always True.
        If neither relation is in the composition table → allow
        (conservative: only block known-bad patterns).

        Parameters
        ----------
        prev_relation : str | None  Relation at previous hop
        curr_relation : str         Relation at current hop
        """
        if prev_relation is None:
            return True

        # If neither is in the composition table, don't constrain
        pattern = (prev_relation, curr_relation)
        known_patterns = {p[0] for p in _VALID_COMPOSITIONS} | \
                         {p[1] for p in _VALID_COMPOSITIONS}

        if prev_relation not in known_patterns and \
           curr_relation not in known_patterns:
            return True

        return pattern in _VALID_COMPOSITIONS

    # ------------------------------------------------------------------
    # Combined admission check
    # ------------------------------------------------------------------

    def is_admissible(
        self,
        relation: str,
        tail_entity_mid: str,
        answer_types: FrozenSet[str],
        hop: int,
        max_hop: int,
        prev_relation: Optional[str] = None,
    ) -> bool:
        """
        Full symbolic admission check for a candidate edge.

        A candidate (relation, tail_entity) is admitted iff it
        passes ALL three symbolic gates.

        This replaces the product cosine score entirely.
        No embeddings, no forward passes, no threshold tuning.

        Parameters
        ----------
        relation        : str    Freebase relation string
        tail_entity_mid : str    MID of the tail/object entity
        answer_types    : frozenset  From infer_answer_types()
        hop             : int    Current hop depth (1-indexed)
        max_hop         : int    Total path depth L
        prev_relation   : str | None  Previous hop's relation (for composition)
        """
        # Gate 1: answer type constraint at terminal hop
        if not self.type_gate(tail_entity_mid, answer_types, hop, max_hop):
            return False

        # Gate 2: property range constraint
        if not self.range_gate(relation, tail_entity_mid):
            return False

        # Gate 3: relation composition constraint
        if not self.composition_gate(prev_relation, relation):
            return False

        return True


# ---------------------------------------------------------------------------
# SIR type component helper
# ---------------------------------------------------------------------------

def compute_type_irrelevance(
    paths: List[Tuple],
    oracle: TypeOracle,
    answer_types: FrozenSet[str],
    max_hop: int,
) -> float:
    """
    Compute SIR*_type for a set of candidate paths.

    SIR*_type = fraction of paths whose terminal entity type
    is incompatible with the inferred answer types.

    Parameters
    ----------
    paths        : list of path tuples [(e0, r1, e1, ..., rh, eh), ...]
    oracle       : TypeOracle instance
    answer_types : frozenset from oracle.infer_answer_types()
    max_hop      : int
    """
    if not paths or not answer_types:
        return 0.0

    n_irrelevant = 0
    for path in paths:
        # Terminal entity is the last element
        terminal = path[-1]
        terminal_mid = terminal if isinstance(terminal, str) else terminal.get("id", "")
        if not oracle.type_gate(terminal_mid, answer_types, max_hop, max_hop):
            n_irrelevant += 1

    return n_irrelevant / len(paths)
