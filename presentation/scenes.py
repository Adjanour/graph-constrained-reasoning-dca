from manim import *
from manim_slides import Slide


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
KG_NODE = "#4ECDC4"
KG_EDGE = "#95E1D3"
QUERY_COLOR = "#F38181"
RELEVANT = "#4ECDC4"
IRRELEVANT = "#FF6B6B"
TRIE_NODE = "#A8E6CF"
PRUNED = "#FF6B6B"
DCA_COLOR = "#556270"
ACCENT = "#FFE66D"
BG_DARK = "#1a1a2e"


# ---------------------------------------------------------------------------
# Scene 0: Title
# ---------------------------------------------------------------------------
class IntroScene(Slide):
    def construct(self):
        # Background
        self.camera.background_color = BG_DARK

        # Title
        title = Text("Dynamic Context-Aware Trie", font_size=48, color=KG_NODE, weight=BOLD)
        subtitle = Text("(DCA-Trie)", font_size=36, color=WHITE)
        author = Text(
            "Bernard\nDepartment of Computer Science\nUniversity of Mines and Technology",
            font_size=20, color=GRAY
        )

        title.move_to(UP * 1.5)
        subtitle.next_to(title, DOWN, buff=0.4)
        author.next_to(subtitle, DOWN, buff=0.6)

        self.play(Write(title), run_time=1.2)
        self.play(Write(subtitle), run_time=0.6)
        self.play(FadeIn(author), run_time=0.5)
        self.next_slide()

        # Problem statement
        problem = VGroup(
            Text("Problem:", font_size=28, color=IRRELEVANT, weight=BOLD),
            Text("Static KG-Tries include irrelevant paths", font_size=24, color=WHITE),
            Text("→ 14.5% of trie paths are semantically irrelevant", font_size=20, color=GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        solution = VGroup(
            Text("Solution:", font_size=28, color=RELEVANT, weight=BOLD),
            Text("Dynamic pruning at each decoding step", font_size=24, color=WHITE),
            Text("→ Only relevant paths survive generation", font_size=20, color=GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        content = VGroup(problem, solution).arrange(DOWN, buff=0.8)
        content.move_to(ORIGIN)

        self.play(Write(problem), run_time=1)
        self.next_slide()
        self.play(Write(solution), run_time=1)
        self.next_slide()
        self.wait(2)


# ---------------------------------------------------------------------------
# Scene 1: Knowledge Graph
# ---------------------------------------------------------------------------
class KGScene(Slide):
    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("Knowledge Graph", font_size=40, color=KG_NODE, weight=BOLD)
        subtitle = Text("Example: Freebase subgraph for question answering", font_size=18, color=GRAY)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(subtitle))
        self.next_slide()

        # KG nodes with better layout
        entities = {
            "Barack Obama": (-3.5, 1.5, 0),
            "USA": (0, 2.5, 0),
            "Harvard Law": (3.5, 1.5, 0),
            "Kenya": (-3.5, -1, 0),
            "Ann Dunham": (0, -2, 0),
            "Constitution": (3.5, -1, 0),
            "President": (0, 0.5, 0),
        }

        node_mobjects = {}
        for name, pos in entities.items():
            circle = Circle(radius=0.45, color=KG_NODE, fill_opacity=0.25, stroke_width=2)
            circle.move_to(pos)
            label = Text(name, font_size=13, color=WHITE).move_to(pos)
            node_mobjects[name] = (circle, label)
            self.play(GrowFromCenter(circle), Write(label), run_time=0.25)

        # Create edges with curved lines for clarity
        edges = [
            ("Barack Obama", "USA", "born_in"),
            ("Barack Obama", "Harvard Law", "educated_at"),
            ("Barack Obama", "President", "held_position"),
            ("Barack Obama", "Kenya", "heritage"),
            ("Barack Obama", "Ann Dunham", "mother"),
            ("President", "USA", "of"),
            ("President", "Constitution", "sworn_under"),
        ]

        for src, dst, rel in edges:
            src_pos = node_mobjects[src][0].get_center()
            dst_pos = node_mobjects[dst][0].get_center()
            # Curved edge
            mid = (src_pos + dst_pos) / 2
            offset = UP * 0.2 if src_pos[1] != dst_pos[1] else LEFT * 0.2
            curve = CubicBezier(
                src_pos, src_pos + offset,
                dst_pos + offset, dst_pos,
                color=KG_EDGE, stroke_width=1.5
            )
            rel_label = Text(rel, font_size=9, color=GRAY_B).move_to(mid + offset * 0.5)
            self.play(Create(curve), Write(rel_label), run_time=0.15)

        # Question
        q_box = RoundedRectangle(width=8, height=0.8, corner_radius=0.2, color=QUERY_COLOR, fill_opacity=0.15)
        q_box.to_edge(DOWN, buff=0.5)
        q_text = Text("Q: Who was the president born in Kenya?", font_size=22, color=QUERY_COLOR)
        q_text.move_to(q_box.get_center())
        self.play(Create(q_box), Write(q_text))
        self.next_slide()

        # Highlight relevant path
        highlight = VGroup(
            node_mobjects["Barack Obama"][0],
            node_mobjects["President"][0],
            node_mobjects["Kenya"][0],
        )
        for h in highlight:
            h.set_color(RELEVANT)
            h.set_fill(RELEVANT, opacity=0.4)
        self.wait(2)


# ---------------------------------------------------------------------------
# Scene 2: GCR Static Trie
# ---------------------------------------------------------------------------
class GCRStaticScene(Slide):
    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("GCR: Static Trie Construction", font_size=36, color=KG_NODE, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Root
        root = Circle(radius=0.35, color=TRIE_NODE, fill_opacity=0.3, stroke_width=2)
        root.move_to(UP * 2)
        root_label = Text("ROOT", font_size=14, color=WHITE).move_to(root.get_center())
        self.play(GrowFromCenter(root), Write(root_label))

        # Level 1
        hop1 = Circle(radius=0.35, color=TRIE_NODE, fill_opacity=0.3, stroke_width=2)
        hop1.move_to(UP * 1 + LEFT * 2.5)
        hop1_label = Text("Barack\nObama", font_size=11, color=WHITE).move_to(hop1.get_center())
        edge1 = Line(root.get_bottom(), hop1.get_top(), color=KG_EDGE, stroke_width=2)
        self.play(Create(edge1), GrowFromCenter(hop1), Write(hop1_label))

        # Level 2: predicates (wider layout)
        preds = [
            ("born_in", RELEVANT, LEFT * 4.5),
            ("educated_at", IRRELEVANT, LEFT * 2.5),
            ("held_position", RELEVANT, LEFT * 0.5),
            ("heritage", IRRELEVANT, RIGHT * 1.5),
            ("mother", IRRELEVANT, RIGHT * 3.5),
        ]

        pred_nodes = []
        for name, color, x_offset in preds:
            pred = RoundedRectangle(
                width=1.4, height=0.4, corner_radius=0.1,
                color=color, fill_opacity=0.25, stroke_width=1.5
            )
            pred.move_to(UP * 0 + x_offset)
            pred_label = Text(name, font_size=10, color=color).move_to(pred.get_center())
            edge = Line(hop1.get_bottom(), pred.get_top(), color=KG_EDGE, stroke_width=1)
            pred_nodes.append((pred, pred_label, edge, color))
            self.play(Create(edge), GrowFromCenter(pred), Write(pred_label), run_time=0.25)

        self.next_slide()

        # Level 3: objects
        objects = [
            ("USA", RELEVANT, LEFT * 4.5),
            ("Harvard", IRRELEVANT, LEFT * 2.5),
            ("President", RELEVANT, LEFT * 0.5),
            ("Kenya", IRRELEVANT, RIGHT * 1.5),
            ("Ann Dunham", IRRELEVANT, RIGHT * 3.5),
        ]

        obj_nodes = []
        for (pred, _, pred_x, _), (name, color, x_offset) in zip(pred_nodes, objects):
            obj = Circle(radius=0.28, color=color, fill_opacity=0.25, stroke_width=1.5)
            obj.move_to(DOWN * 1 + x_offset)
            obj_label = Text(name, font_size=10, color=color).move_to(obj.get_center())
            edge = Line(pred.get_bottom(), obj.get_top(), color=KG_EDGE, stroke_width=1)
            obj_nodes.append((obj, obj_label, color))
            self.play(Create(edge), GrowFromCenter(obj), Write(obj_label), run_time=0.2)

        # Highlight irrelevant
        self.next_slide()

        irrelevant_count = sum(1 for _, _, _, c in pred_nodes if c == IRRELEVANT)
        total = len(pred_nodes)

        # Flash irrelevant nodes
        for pred, label, _, color in pred_nodes:
            if color == IRRELEVANT:
                self.play(pred.animate.set_fill(IRRELEVANT, opacity=0.6), run_time=0.15)

        for obj, label, color in obj_nodes:
            if color == IRRELEVANT:
                self.play(obj.animate.set_fill(IRRELEVANT, opacity=0.6), run_time=0.15)

        # SIR explanation
        sir_box = RoundedRectangle(width=7, height=1.2, corner_radius=0.2, color=IRRELEVANT, fill_opacity=0.1)
        sir_box.to_edge(DOWN, buff=0.3)
        sir_text = VGroup(
            Text("Semantic Irrelevance Ratio (SIR)", font_size=20, color=IRRELEVANT, weight=BOLD),
            Text(f"{irrelevant_count}/{total} paths are irrelevant = {irrelevant_count/total*100:.1f}%", font_size=18, color=WHITE),
            Text("These paths waste compute and can cause hallucination", font_size=16, color=GRAY),
        ).arrange(DOWN, buff=0.15)
        sir_text.move_to(sir_box.get_center())
        self.play(Create(sir_box), Write(sir_text))
        self.next_slide()
        self.wait(2)


# ---------------------------------------------------------------------------
# Scene 3: DCA-Trie v2 Dynamic Pruning
# ---------------------------------------------------------------------------
class DCATrieScene(Slide):
    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("DCA-Trie v2: Dynamic Pruning", font_size=36, color=RELEVANT, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Question context
        q_box = RoundedRectangle(width=7, height=0.6, corner_radius=0.15, color=QUERY_COLOR, fill_opacity=0.15)
        q_box.move_to(UP * 2.2)
        q_text = Text('Q: "Who was the president born in Kenya?"', font_size=18, color=QUERY_COLOR)
        q_text.move_to(q_box.get_center())
        self.play(Create(q_box), Write(q_text))

        # Root
        root = Circle(radius=0.3, color=TRIE_NODE, fill_opacity=0.3, stroke_width=2)
        root.move_to(UP * 1)
        root_label = Text("ROOT", font_size=12, color=WHITE).move_to(root.get_center())
        self.play(GrowFromCenter(root), Write(root_label))

        # Hop 1: Barack Obama
        hop1 = Circle(radius=0.3, color=TRIE_NODE, fill_opacity=0.3, stroke_width=2)
        hop1.move_to(UP * 0 + LEFT * 3)
        hop1_label = Text("Barack\nObama", font_size=10, color=WHITE).move_to(hop1.get_center())
        edge1 = Line(root.get_bottom(), hop1.get_top(), color=KG_EDGE, stroke_width=2)
        self.play(Create(edge1), GrowFromCenter(hop1), Write(hop1_label))

        # Step 1: Generate "born_in"
        step1_box = RoundedRectangle(width=5, height=0.5, corner_radius=0.1, color=RELEVANT, fill_opacity=0.15)
        step1_box.move_to(DOWN * 0.8)
        step1 = Text("Step 1: Generate 'born_in' (relevant to question)", font_size=16, color=RELEVANT)
        step1.move_to(step1_box.get_center())
        self.play(Create(step1_box), Write(step1))

        born_in = RoundedRectangle(width=1.3, height=0.35, corner_radius=0.1, color=RELEVANT, fill_opacity=0.3, stroke_width=1.5)
        born_in.move_to(UP * -1 + LEFT * 3)
        born_in_label = Text("born_in", font_size=10, color=RELEVANT).move_to(born_in.get_center())
        edge_b = Line(hop1.get_bottom(), born_in.get_top(), color=KG_EDGE, stroke_width=2)
        self.play(Create(edge_b), GrowFromCenter(born_in), Write(born_in_label))

        # Show faded alternatives
        faded_preds = [
            ("educated_at", LEFT * 1),
            ("held_position", RIGHT * 1),
            ("heritage", RIGHT * 3),
        ]
        faded_nodes = []
        for name, x in faded_preds:
            faded = RoundedRectangle(width=1.1, height=0.3, corner_radius=0.1, color=GRAY, fill_opacity=0.05, stroke_width=1)
            faded.move_to(UP * -1 + x)
            faded_label = Text(name, font_size=8, color=GRAY_B).move_to(faded.get_center())
            faded_nodes.append((faded, faded_label))
            self.play(GrowFromCenter(faded), Write(faded_label), run_time=0.12)

        self.next_slide()

        # Step 2: Generate "Kenya"
        step2 = Text("Step 2: Generate 'Kenya' (range-compatible)", font_size=16, color=RELEVANT)
        step2.move_to(step1_box.get_center())
        self.play(Transform(step1, step2))

        kenya = Circle(radius=0.25, color=RELEVANT, fill_opacity=0.3, stroke_width=1.5)
        kenya.move_to(DOWN * 2 + LEFT * 3)
        kenya_label = Text("Kenya", font_size=10, color=RELEVANT).move_to(kenya.get_center())
        usa = Circle(radius=0.25, color=GRAY, fill_opacity=0.05, stroke_width=1)
        usa.move_to(DOWN * 2 + LEFT * 1)
        usa_label = Text("USA", font_size=10, color=GRAY_B).move_to(usa.get_center())
        edge_k = Line(born_in.get_bottom(), kenya.get_top(), color=KG_EDGE, stroke_width=2)
        edge_u = Line(born_in.get_bottom(), usa.get_top(), color=GRAY, stroke_width=1)
        self.play(Create(edge_k), GrowFromCenter(kenya), Write(kenya_label))
        self.play(Create(edge_u), GrowFromCenter(usa), Write(usa_label))

        self.next_slide()

        # Step 3: Final answer
        step3 = Text("Step 3: Extract answer from constrained path", font_size=16, color=RELEVANT)
        step3.move_to(step1_box.get_center())
        self.play(Transform(step1, step3))

        # Fade out irrelevant
        for faded, faded_label in faded_nodes:
            self.play(FadeOut(faded), FadeOut(faded_label), run_time=0.08)
        self.play(FadeOut(usa), FadeOut(usa_label))

        # Show result
        result_box = RoundedRectangle(width=6, height=0.8, corner_radius=0.2, color=RELEVANT, fill_opacity=0.15)
        result_box.move_to(DOWN * 2.5)
        result = Text("Path: Barack Obama → born_in → Kenya", font_size=18, color=RELEVANT, weight=BOLD)
        result.move_to(result_box.get_center())
        self.play(Create(result_box), Write(result))
        self.next_slide()
        self.wait(2)


# ---------------------------------------------------------------------------
# Scene 4: Algorithm Overview
# ---------------------------------------------------------------------------
class AlgorithmScene(Slide):
    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("DCA-Trie Algorithm", font_size=36, color=ACCENT, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Algorithm steps
        steps = [
            ("1", "Build initial trie from first-hop gated neighbours", RELEVANT),
            ("2", "Generate next token sequence with trie constraint", KG_NODE),
            ("3", "Commit generated entity, update prompt", ACCENT),
            ("4", "Expand trie from committed entity's neighbours", RELEVANT),
            ("5", "Repeat until path complete or max hops", KG_NODE),
        ]

        for i, (num, text, color) in enumerate(steps):
            y = UP * 1.5 - i * 0.7

            # Step number
            num_circle = Circle(radius=0.25, color=color, fill_opacity=0.3, stroke_width=2)
            num_circle.move_to(LEFT * 4 + y)
            num_text = Text(num, font_size=16, color=color, weight=BOLD).move_to(num_circle.get_center())

            # Step text
            step_text = Text(text, font_size=18, color=WHITE)
            step_text.move_to(LEFT * 1 + y)

            # Arrow (except last)
            if i < len(steps) - 1:
                arrow = Arrow(
                    LEFT * 4 + y + DOWN * 0.3,
                    LEFT * 4 + y + DOWN * 0.6,
                    color=GRAY, stroke_width=1.5, max_tip_length_to_length_ratio=0.15
                )
                self.play(Create(arrow), run_time=0.1)

            self.play(GrowFromCenter(num_circle), Write(num_text), Write(step_text), run_time=0.4)

        self.next_slide()

        # Key insight
        insight_box = RoundedRectangle(width=8, height=1, corner_radius=0.2, color=ACCENT, fill_opacity=0.1)
        insight_box.move_to(DOWN * 2.5)
        insight = VGroup(
            Text("Key Insight:", font_size=20, color=ACCENT, weight=BOLD),
            Text("Trie expands dynamically at each entity commit,", font_size=16, color=WHITE),
            Text("conditioning on question + partial generation", font_size=16, color=WHITE),
        ).arrange(DOWN, buff=0.1)
        insight.move_to(insight_box.get_center())
        self.play(Create(insight_box), Write(insight))
        self.next_slide()
        self.wait(2)


# ---------------------------------------------------------------------------
# Scene 5: Comparison
# ---------------------------------------------------------------------------
class ComparisonScene(Slide):
    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("Results: Trie Size Comparison", font_size=36, color=ACCENT, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Bar chart comparison
        bar_width = 1.5
        max_height = 3

        # GCR bar
        gcr_bar = Rectangle(width=bar_width, height=max_height, color=IRRELEVANT, fill_opacity=0.4, stroke_width=2)
        gcr_bar.move_to(LEFT * 3 + DOWN * 0.5)
        gcr_bar.align_to(DOWN * 2, UP)
        gcr_label = Text("GCR\n(Static)", font_size=16, color=IRRELEVANT, weight=BOLD)
        gcr_label.next_to(gcr_bar, DOWN, buff=0.2)
        gcr_value = Text("100%", font_size=20, color=IRRELEVANT, weight=BOLD)
        gcr_value.next_to(gcr_bar, UP, buff=0.1)
        self.play(GrowFromEdge(gcr_bar, DOWN), Write(gcr_label), Write(gcr_value))

        # DCA-Trie bar (85.5% height)
        dca_height = max_height * 0.855
        dca_bar = Rectangle(width=bar_width, height=dca_height, color=RELEVANT, fill_opacity=0.4, stroke_width=2)
        dca_bar.move_to(RIGHT * 3 + DOWN * 0.5)
        dca_bar.align_to(DOWN * 2, UP)
        dca_label = Text("DCA-Trie\n(Dynamic)", font_size=16, color=RELEVANT, weight=BOLD)
        dca_label.next_to(dca_bar, DOWN, buff=0.2)
        dca_value = Text("85.5%", font_size=20, color=RELEVANT, weight=BOLD)
        dca_value.next_to(dca_bar, UP, buff=0.1)
        self.play(GrowFromEdge(dca_bar, DOWN), Write(dca_label), Write(dca_value))

        # Reduction arrow
        reduction = VGroup(
            Text("14.5%", font_size=28, color=YELLOW, weight=BOLD),
            Text("reduction", font_size=16, color=YELLOW),
        ).arrange(DOWN, buff=0.1)
        reduction.move_to(ORIGIN)
        self.play(Write(reduction))

        self.next_slide()

        # Metrics
        metrics = VGroup(
            Text("Metrics:", font_size=20, color=ACCENT, weight=BOLD),
            Text("• SIR: 14.5% (type: 10.6%, traj: 3.8%)", font_size=16, color=WHITE),
            Text("• Type FNR: 3.3% (< 5% target ✓)", font_size=16, color=RELEVANT),
            Text("• Range FNR: 2.9%", font_size=16, color=WHITE),
            Text("• Structural faithfulness: 100%", font_size=16, color=RELEVANT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        metrics.move_to(DOWN * 2.2)
        self.play(Write(metrics))
        self.next_slide()
        self.wait(2)
