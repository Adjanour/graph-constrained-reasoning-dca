"""
intent_parser.py — ORT-style ontology-guided intent parser for DCA-Trie.

Replaces the fragile regex-based answer type gate (Gate 1) with
a lightweight LLM call that maps a question to its target Freebase
type label before decoding begins.

Usage:
    parser = IntentParser(model, oracle)
    answer_types = parser.parse("Which film directed by James Cameron?")
    # Returns: frozenset({"Film"})

Architecture (ORT-inspired, Figure 3):
    1. Single-shot LLM call to classify question intent → aim_label
    2. Map aim_label to Freebase type set via oracle schema
    3. Fall back to regex if LLM is unavailable or fails
    4. Cache results by question hash
"""

import logging
import re
from typing import FrozenSet, Optional, Set, Dict

logger = logging.getLogger("type_oracle")

INTENT_CLASSIFICATION_PROMPT = """Given a question, identify the single most specific Freebase type of the expected answer.

Question: "{question}"

Choose the best matching type from this list:
{type_list}

Output ONLY the type name, nothing else. For example:
Person
Location
Film"""

# Mapping from LLM-output aim labels to Freebase type frozensets
AIM_TO_FREEBASE_TYPES: Dict[str, FrozenSet[str]] = {
    "Person": frozenset({"Person", "Deceased Person", "Politician", "Musical Artist",
                         "Author", "Film director", "Film actor", "TV Actor",
                         "Scientist", "Athlete", "Fictional Character"}),
    "Location": frozenset({"Location", "Country", "City/Town/Village", "US State",
                           "Mountain", "River", "Island", "Body of Water",
                           "Geographical Feature", "Administrative Division"}),
    "Organization": frozenset({"Organization", "Company", "Educational Institution",
                                "Government Agency", "Sports Team", "Broadcaster",
                                "Non-profit Organization", "Military Unit"}),
    "CreativeWork": frozenset({"Film", "Book", "TV Program", "Written Work",
                                "Musical Work", "Album", "Song", "Artwork",
                                "Video Game", "Software"}),
    "Date": frozenset({"Date", "Year", "Decade", "Century", "Time Period",
                        "Holiday", "Season"}),
    "Language": frozenset({"Language", "Dialect", "Programming Language"}),
    "Award": frozenset({"Award", "Prize", "Honor", "Military Award",
                         "Film Award", "Music Award"}),
    "Profession": frozenset({"Profession", "Job", "Occupation", "Field of Study",
                              "Medical Specialty", "Sport"}),
    "Event": frozenset({"Event", "Sports Event", "Music Event", "Conference",
                         "Military Conflict", "Natural Disaster", "Festival"}),
    "Product": frozenset({"Product", "Brand", "Model", "Drug", "Device"}),
    "Species": frozenset({"Species", "Animal", "Plant", "Fungus", "Bacteria",
                           "Biological Classification"}),
    "Building": frozenset({"Building", "Structure", "Bridge", "Stadium",
                            "Airport", "Hotel", "Hospital", "Museum"}),
}


class IntentParser:
    """
    ORT-style ontology-guided intent parser.
    
    Uses a single lightweight LLM call to classify the question's
    expected answer type, replacing the regex-based approach.
    """

    def __init__(self, model=None, oracle=None, cache_size: int = 1024):
        self._model = model
        self._oracle = oracle
        self._cache: Dict[str, FrozenSet[str]] = {}
        self._cache_size = cache_size
        self._stats = {"llm_calls": 0, "cache_hits": 0, "fallbacks": 0}

    @property
    def stats(self) -> Dict:
        return dict(self._stats)

    def parse(self, question: str) -> FrozenSet[str]:
        cache_key = question.lower().strip()

        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        result = self._parse_impl(question)

        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[cache_key] = result
        return result

    def _parse_impl(self, question: str) -> FrozenSet[str]:
        if self._model is not None:
            aim_label = self._classify_with_llm(question)
            if aim_label and aim_label in AIM_TO_FREEBASE_TYPES:
                self._stats["llm_calls"] += 1
                return AIM_TO_FREEBASE_TYPES[aim_label]

        self._stats["fallbacks"] += 1
        return self._fallback_regex(question)

    def _classify_with_llm(self, question: str) -> Optional[str]:
        type_list = "\n".join(f"  - {k}" for k in sorted(AIM_TO_FREEBASE_TYPES.keys()))
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            question=question, type_list=type_list
        )

        try:
            llm_input = self._model.prepare_model_prompt(prompt)
            import torch
            inputs = self._model.tokenizer(
                llm_input, return_tensors="pt", add_special_tokens=False
            )
            input_ids = inputs.input_ids.to(self._model.model.device)
            attention_mask = inputs.attention_mask.to(self._model.model.device)

            with torch.no_grad():
                res = self._model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._model.tokenizer.eos_token_id,
                )

            output = self._model.tokenizer.decode(
                res.sequences[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            for label in sorted(AIM_TO_FREEBASE_TYPES.keys(), key=len, reverse=True):
                if label.lower() in output.lower():
                    return label

        except Exception as e:
            logger.debug("LLM intent classification failed: %s", e)

        return None

    def _fallback_regex(self, question: str) -> FrozenSet[str]:
        if self._oracle is not None:
            return self._oracle.infer_answer_types(question)
        return frozenset()
