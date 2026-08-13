"""
lazy_constraint.py — Lazily materialised, state-conditioned KG constraint.

``LazyGraphConstraint`` duck-types :class:`src.trie.MarisaTrie` — it exposes a
single ``get(prefix_token_ids) -> List[int]`` method — so it drops into the
existing :class:`GraphConstrainedDecoding` with no changes to
``allowed_tokens_fn`` or ``generate_sentence``.  Baseline, v1 and this share
the *same* decoding call; only the constraint object differs.

Contrast with the other two constraint builders:

- **v1 (static)**  DFS enumerates every path up to L hops, TypeOracle filters
  them, the whole set is tokenised into one trie before decoding starts.
- **v2 (rebuilt)** One trie per hop per beam, one ``generate()`` call per hop.
  Beam scores come from different, separately-normalised supports, and path
  length is fixed by the Python loop rather than chosen by the model.
- **here (lazy)**  One trie, one ``generate()`` call, one probability space —
  but the trie's nodes are materialised on demand from the graph as the beams
  reach them.  Nothing beyond the visited frontier is ever enumerated.

The gates move with the frontier, which is what makes them better informed
than v1's:

- ``range_gate(rel, tail)`` admits an outgoing edge.
- ``type_gate(entity, answer_types, hop, hop)`` decides whether the path may
  *stop* at the current entity — passing ``hop`` as both arguments forces the
  terminal branch, so the question is "is this a legal answer?" rather than
  v1's "is this the right type at exactly depth L?".

The model still owns termination: ``</PATH>`` is offered as a legal token at
every entity that passes the type gate, exactly as it is in the static trie.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from src.trie import Trie

from utils import PATH_START, PATH_END, logger

TokenSeq = Tuple[int, ...]


@dataclass(frozen=True)
class Anchor:
    """A confirmed position in the path: everything up to an entity boundary."""

    text: str                  # path text so far, without the <PATH> sentinel
    entity: Optional[str]      # current entity; None at the root
    hop: int                   # hops committed so far
    visited: FrozenSet[str]    # entities already on this path (cycle guard)


@dataclass
class Frontier:
    """The materialised continuations available from one anchor."""

    trie: Trie
    # completed continuation -> anchor that follows it; the close option maps
    # to None because decoding leaves constrained mode once </PATH> is emitted.
    complete: Dict[TokenSeq, Optional[Anchor]]
    n_candidates: int


class LazyGraphConstraint:
    """Graph-backed constraint that materialises trie nodes on demand.

    Parameters
    ----------
    tokenizer :
        HuggingFace tokenizer, used to render candidate strings into token ids.
    nx_graph : networkx.DiGraph
        Question subgraph, edges carrying a ``relation`` attribute.
    start_entities : list of str
        Linked question entities; the roots of every legal path.
    oracle : TypeOracle
        Supplies ``range_gate`` and ``type_gate``.
    answer_types : frozenset of str
        Output of ``oracle.infer_answer_types(question)``.
    max_hops : int
        Hop budget, equivalent to v1's ``index_len``.
    gates_enabled : bool
        If False, admit every edge and allow the path to close anywhere —
        lazy expansion with no semantic pruning.  This is the control that
        isolates the gates: with gates off the admitted language is exactly
        the static baseline trie's, so any accuracy difference against
        ``GCR_Baseline`` comes from the constraint mechanism rather than from
        the gates.
    block_cycles : bool
        Forbid revisiting an entity already on the path.  Off by default: the
        static DFS in ``graph_utils.dfs`` permits revisits, and Freebase name
        collapsing creates genuine self-loops (the species "Rabbit" and the
        character "Rabbit" are one node), so enabling this changes the
        admitted language rather than just tightening it.
    """

    def __init__(
        self,
        tokenizer,
        nx_graph,
        start_entities: Sequence[str],
        oracle,
        answer_types: FrozenSet[str],
        max_hops: int,
        gates_enabled: bool = True,
        block_cycles: bool = False,
    ):
        self.tokenizer = tokenizer
        self.graph = nx_graph
        self.start_entities = [e for e in start_entities if e in nx_graph]
        self.oracle = oracle
        self.answer_types = answer_types
        self.max_hops = max_hops
        self.gates_enabled = gates_enabled
        self.block_cycles = block_cycles

        self._root = Anchor(text="", entity=None, hop=0, visited=frozenset())

        # Memo tables.  Both are per-question and bounded by the number of
        # distinct partial paths the beams actually walk — at most
        # beam_size * max_hops entries in practice.
        self._frontiers: Dict[Anchor, Frontier] = {}
        self._anchor_of: Dict[TokenSeq, Anchor] = {(): self._root}
        self._anchor_lengths: Set[int] = {0}

        # Instrumentation: how much of the graph we avoided touching.
        self.n_frontier_builds = 0
        self.n_candidates_materialised = 0

    # ------------------------------------------------------------------
    # MarisaTrie-compatible entry point
    # ------------------------------------------------------------------

    def get(self, prefix_sequence: List[int]) -> List[int]:
        """Return the legal next token ids for *prefix_sequence*.

        ``prefix_sequence`` is the slice from the open ``<PATH>`` token
        onwards, as sliced by ``GraphConstrainedDecoding.allowed_tokens_fn``.
        """
        prefix = tuple(prefix_sequence)
        allowed: Set[int] = set()
        seen: Set[Anchor] = set()
        pending = self._anchor_chain(prefix)

        while pending:
            anchor = pending.pop()
            if anchor in seen:
                continue
            seen.add(anchor)

            frontier = self._frontier(anchor)
            allowed |= set(frontier.trie.get(prefix_sequence))

            # The prefix completing one continuation does not exhaust this
            # frontier: a candidate's tokens can be a proper prefix of a
            # sibling's ("Characters" against "Characters with this
            # condition").  Union in the frontier this opens rather than
            # switching to it, or those siblings become unreachable.
            if prefix in frontier.complete:
                next_anchor = frontier.complete[prefix]
                if next_anchor is not None:
                    self._anchor_of.setdefault(prefix, next_anchor)
                    self._anchor_lengths.add(len(prefix))
                    pending.append(next_anchor)

        if not allowed:
            logger.warning("lazy constraint: no continuation for a %d-token prefix",
                           len(prefix))
        return list(allowed)

    # ------------------------------------------------------------------
    # Anchor resolution
    # ------------------------------------------------------------------

    def _anchor_chain(self, prefix: TokenSeq) -> List[Anchor]:
        """Every confirmed anchor lying strictly before *prefix*, root first.

        For the same collision reason, no single anchor is authoritative: the
        prefix may still be inside a shallower frontier's candidate as well as
        past a deeper one's.
        """
        chain = [self._root]
        for length in sorted(self._anchor_lengths):
            if length == 0 or length >= len(prefix):
                continue
            anchor = self._anchor_of.get(prefix[:length])
            if anchor is not None:
                chain.append(anchor)
        return chain

    # ------------------------------------------------------------------
    # Frontier materialisation
    # ------------------------------------------------------------------

    def _frontier(self, anchor: Anchor) -> Frontier:
        cached = self._frontiers.get(anchor)
        if cached is not None:
            return cached

        # (continuation text, anchor that follows it) pairs.
        options: List[Tuple[str, Optional[Anchor]]] = []

        if anchor.entity is None:
            for entity in self.start_entities:
                visited = anchor.visited | {entity} if self.block_cycles else anchor.visited
                for rel, nbr in self._gated_edges(entity, visited):
                    text = f"{entity} -> {rel} -> {nbr}"
                    options.append((text, self._advance(anchor, text, nbr, visited)))
        else:
            if anchor.hop < self.max_hops:
                for rel, nbr in self._gated_edges(anchor.entity, anchor.visited):
                    text = f" -> {rel} -> {nbr}"
                    options.append((text, self._advance(anchor, text, nbr, anchor.visited)))

        # The close option.  Offered whenever the current entity is a legal
        # answer — and unconditionally when nothing else is left, so the
        # constraint can never hand back an empty set and silently fall back
        # to unconstrained decoding.
        if anchor.entity is not None and (self._may_close(anchor) or not options):
            options.append((PATH_END, None))

        sequences: List[List[int]] = []
        complete: Dict[TokenSeq, Optional[Anchor]] = {}

        for text, next_anchor in options:
            # Tokenise the whole block from <PATH> every time.  The anchor
            # carries its own text, so this never round-trips through
            # decode(); token boundaries stay identical to what the model was
            # masked with on the previous step.
            full = f"{PATH_START}{anchor.text}{text}"
            ids = self.tokenizer(full, add_special_tokens=False).input_ids
            if next_anchor is None:
                ids = ids + [self.tokenizer.eos_token_id]
            sequences.append(ids)
            complete[tuple(ids if next_anchor is not None else ids[:-1])] = next_anchor

        frontier = Frontier(
            trie=Trie(sequences),
            complete=complete,
            n_candidates=len(options),
        )
        self._frontiers[anchor] = frontier
        self.n_frontier_builds += 1
        self.n_candidates_materialised += len(options)
        return frontier

    def _advance(
        self, anchor: Anchor, text: str, entity: str, visited: FrozenSet[str]
    ) -> Anchor:
        return Anchor(
            text=anchor.text + text,
            entity=entity,
            hop=anchor.hop + 1,
            visited=frozenset(visited | {entity}) if self.block_cycles else visited,
        )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _gated_edges(self, head: str, visited: FrozenSet[str]):
        """Outgoing edges of *head* that survive the range gate."""
        if head not in self.graph:
            return
        for nbr in self.graph.neighbors(head):
            if self.block_cycles and nbr in visited:
                continue
            rel = self.graph[head][nbr]["relation"]
            if self.gates_enabled and not self.oracle.range_gate(rel, nbr):
                continue
            yield rel, nbr

    def _may_close(self, anchor: Anchor) -> bool:
        """Whether the path may legally terminate at *anchor*.

        ``hop`` is passed as both arguments so ``type_gate`` takes its
        terminal branch: the question is whether this entity is an admissible
        *answer*, not whether it sits at depth ``max_hops``.
        """
        if not self.gates_enabled:
            return True
        return self.oracle.type_gate(
            anchor.entity, self.answer_types, anchor.hop, anchor.hop
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Materialisation cost, for the memory/scalability comparison."""
        return {
            "frontier_builds": self.n_frontier_builds,
            "candidates_materialised": self.n_candidates_materialised,
            "anchors_visited": len(self._anchor_of),
        }
