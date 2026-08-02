from manim import *
import numpy as np

config.pixel_height = 2160
config.pixel_width = 3840
config.frame_rate = 30


class PipeScene(Scene):
    camera.background_color = "#000000"

    def dot(self, radius=0.18, color=GREY_B):
        return Dot(radius=radius, color=color, fill_opacity=0.9)

    def arrow(self, s, e, color=GREY_B, sw=2, buff=0.25):
        return Arrow(s, e, buff=buff, stroke_width=sw,
                     color=color, max_tip_length_to_length_ratio=0.06)


# ──────────────────────────────────────────────────────────
# Scene 1: KG + Path Explosion
# ──────────────────────────────────────────────────────────

class KGPathExplosion(PipeScene):
    def construct(self):
        self.show_kg()
        self.highlight_path()
        self.path_explosion()
        self.conclusion()

    def show_kg(self):
        pos = {
            "Blue_Hawaii":    LEFT * 3.5,
            "Norman_Taurog":  ORIGIN,
            "United_States":  RIGHT * 3.5,
            "Elvis_Presley":  LEFT * 0.5 + UP * 2.5,
            "Hawaii":         LEFT * 4.5 + UP * 2.0,
            "Date_1901":      RIGHT * 2.5 + UP * 2.5,
        }
        colors = {"Blue_Hawaii": BLUE, "Norman_Taurog": BLUE, "United_States": BLUE}
        self.nodes = {}
        for name, p in pos.items():
            c = colors.get(name, GREY_B)
            d = self.dot(color=c)
            d.move_to(p)
            lbl = Text(name.replace("_", " "), font_size=22, color=c)
            lbl.next_to(d, DOWN, buff=0.12)
            self.nodes[name] = VGroup(d, lbl)

        conns = [
            ("Blue_Hawaii", "Norman_Taurog", "film.director"),
            ("Blue_Hawaii", "Elvis_Presley", "film.starring"),
            ("Blue_Hawaii", "Hawaii", "film.location"),
            ("Norman_Taurog", "United_States", "nationality"),
            ("Norman_Taurog", "Date_1901", "date_of_birth"),
        ]
        self.arrows = VGroup()
        self.lbls = VGroup()
        for src, dst, rl in conns:
            s = self.nodes[src][0].get_center()
            e = self.nodes[dst][0].get_center()
            mid = (s + e) / 2
            a = self.arrow(s, e)
            l = Text(rl, font_size=18, color=GREY_B)
            l.move_to(mid + UP * 0.3)
            self.arrows.add(a)
            self.lbls.add(l)

        self.play(LaggedStart(*[FadeIn(n, scale=0.5) for n in self.nodes.values()], lag_ratio=0.12), run_time=2)
        self.play(LaggedStart(*[Create(a) for a in self.arrows], lag_ratio=0.08), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(l) for l in self.lbls], lag_ratio=0.08), run_time=1)
        self.wait(0.5)

    def highlight_path(self):
        nt = self.nodes["Norman_Taurog"]
        us = self.nodes["United_States"]
        a1, l1 = self.arrows[0], self.lbls[0]
        a2, l2 = self.arrows[3], self.lbls[3]
        bh = self.nodes["Blue_Hawaii"]
        glow = self.dot(radius=0.04, color=YELLOW)
        glow.move_to(bh[0].get_center())
        p1 = Line(bh[0].get_center(), nt[0].get_center(), color=YELLOW)
        p2 = Line(nt[0].get_center(), us[0].get_center(), color=YELLOW)
        self.play(
            a1.animate.set_color(YELLOW).set_stroke_width(4),
            l1.animate.set_color(YELLOW),
            nt[0].animate.set_color(YELLOW), nt[1].animate.set_color(YELLOW),
            MoveAlongPath(glow, p1, run_time=1),
        )
        self.play(
            a2.animate.set_color(YELLOW).set_stroke_width(4),
            l2.animate.set_color(YELLOW),
            us[0].animate.set_color(YELLOW), us[1].animate.set_color(YELLOW),
            MoveAlongPath(glow, p2, run_time=1),
        )
        self.wait(1)
        for a in self.arrows: a.set_color(GREY_B).set_stroke_width(2)
        for l in self.lbls: l.set_color(GREY_B)
        for n in self.nodes.values():
            n[0].set_color(GREY_B)
            n[1].set_color(GREY_B)
        bh[0].set_color(BLUE)
        bh[1].set_color(BLUE)

    def path_explosion(self):
        self.play(
            *[FadeOut(v) for v in self.nodes.values()],
            *[FadeOut(a) for a in self.arrows],
            *[FadeOut(l) for l in self.lbls], run_time=0.3,
        )
        c = self.dot(radius=0.22, color=BLUE).move_to(ORIGIN)
        cl = Text("Blue_Hawaii", font_size=28, color=BLUE).next_to(c, DOWN, buff=0.15)
        cg = VGroup(c, cl)
        self.play(FadeIn(cg, scale=0.5), run_time=0.5)
        rays = VGroup()
        for a in np.linspace(0, 2 * PI, 8)[:-1]:
            end = ORIGIN + np.array([np.cos(a) * 2.8, np.sin(a) * 2.8, 0])
            rays.add(self.arrow(ORIGIN, end, color=TEAL, sw=2, buff=0.3))
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.05), run_time=1.2)
        deg = Text("deg ~ 20", font_size=28, color=TEAL).next_to(cg, RIGHT, buff=1.5)
        self.play(Write(deg), run_time=0.4)
        ring2 = VGroup()
        for a in np.linspace(0, 2 * PI, 8)[:-1]:
            start = ORIGIN + np.array([np.cos(a) * 2.8, np.sin(a) * 2.8, 0])
            for da in [-0.4, 0.4]:
                end = start + np.array([np.cos(a + da) * 1.8, np.sin(a + da) * 1.8, 0])
                ring2.add(self.arrow(start, end, color=GREY_C, sw=1.5, buff=0.15))
        self.play(LaggedStart(*[Create(r) for r in ring2], lag_ratio=0.02), run_time=1.5)
        f = MathTex("20^3", "=", "8{,}000", font_size=56, color=RED)
        f.move_to(DOWN * 2)
        rv = Text("only 1 correct", font_size=30, color=RED).next_to(f, DOWN, buff=0.3)
        self.play(Write(f), run_time=0.6)
        self.play(Write(rv), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(f), FadeOut(rv), FadeOut(deg),
                  FadeOut(cg), FadeOut(rays), FadeOut(ring2), run_time=0.3)

    def conclusion(self):
        t = Text("Need a tighter oracle", font_size=48, color=TEAL)
        s = Text("TypeOracle: KG ontology pruning", font_size=32, color=GREY_B)
        s.next_to(t, DOWN, buff=0.3)
        self.play(Write(t), run_time=0.6)
        self.play(FadeIn(s, shift=UP * 0.1), run_time=0.5)
        self.wait(1.5)


# ──────────────────────────────────────────────────────────
# Scene 2: TypeOracle Gates
# ──────────────────────────────────────────────────────────

class TypeOracleGates(PipeScene):
    def construct(self):
        self.type_gate()
        self.clear()
        self.range_gate()
        self.clear()
        self.both_gates()

    def type_gate(self):
        bh = self.dot(color=BLUE).move_to(LEFT * 3.5)
        bhl = Text("Blue_Hawaii", font_size=22, color=BLUE).next_to(bh, DOWN, buff=0.12)
        self.play(FadeIn(VGroup(bh, bhl)), run_time=0.3)
        n_pos = [UP * 1.8, DOWN * 1.8]
        n_names = ["Norman_Taurog\n[Person]", "Hawaii\n[Location]"]
        n_cols = [GREEN, RED]
        neighbors = VGroup()
        edges = VGroup()
        for pos, name, col in zip(n_pos, n_names, n_cols):
            d = self.dot(color=col).move_to(bh.get_center() + pos)
            lbl = Text(name, font_size=18, color=col).next_to(d, DOWN, buff=0.1)
            neighbors.add(VGroup(d, lbl))
            edges.add(self.arrow(bh.get_center(), d.get_center(), color=GREY_B))
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.2), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(n) for n in neighbors], lag_ratio=0.2), run_time=0.6)
        q = Text('"Who directed Blue_Hawaii?"', font_size=30, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(q), run_time=0.5)
        gl = Line(UP * 2.5, DOWN * 2.5, color=BLUE, stroke_width=5).shift(RIGHT * 0.5)
        gtxt = Text("expects: Person", font_size=24, color=BLUE).next_to(gl, UP, buff=0.15)
        self.play(Create(gl), Write(gtxt), run_time=0.5)
        p1 = self.arrow(bh.get_center(), n_pos[0] + bh.get_center(), color=WHITE, sw=3, buff=0.15)
        dot1 = self.dot(radius=0.05, color=WHITE).move_to(bh.get_center())
        m1 = Text("PASS", font_size=22, color=GREEN).next_to(gl, RIGHT, buff=0.3).shift(UP * 1.0)
        self.play(Create(p1), run_time=0.2)
        self.play(MoveAlongPath(dot1, p1), run_time=0.6)
        self.play(FadeOut(dot1), Write(m1), run_time=0.2)
        p2 = self.arrow(bh.get_center(), n_pos[1] + bh.get_center(), color=WHITE, sw=3, buff=0.15)
        dot2 = self.dot(radius=0.05, color=WHITE).move_to(bh.get_center())
        m2 = Text("BLOCK", font_size=22, color=RED).next_to(gl, RIGHT, buff=0.3).shift(DOWN * 1.2)
        self.play(Create(p2), run_time=0.2)
        self.play(MoveAlongPath(dot2, p2), run_time=0.6)
        self.play(FadeOut(dot2), Write(m2), run_time=0.2)
        self.wait(1.5)
        self.play(FadeOut(q), FadeOut(gl), FadeOut(gtxt),
                  FadeOut(m1), FadeOut(m2), FadeOut(edges),
                  FadeOut(neighbors), FadeOut(VGroup(bh, bhl)), run_time=0.3)

    def range_gate(self):
        bh = self.dot(color=BLUE).move_to(LEFT * 3.5)
        bhl = Text("Blue_Hawaii", font_size=22, color=BLUE).next_to(bh, DOWN, buff=0.12)
        self.play(FadeIn(VGroup(bh, bhl)), run_time=0.3)
        n_pos = [UP * 1.8, DOWN * 1.8]
        n_names = ["Norman_Taurog\n[Person]", "Hawaii\n[Location]"]
        n_cols = [GREEN, RED]
        rel = Text("film.director  =>  range: {Person}", font_size=26, color=TEAL).to_edge(UP, buff=0.5)
        self.play(Write(rel), run_time=0.4)
        edges = VGroup()
        neighbors = VGroup()
        for pos, name, col in zip(n_pos, n_names, n_cols):
            d = self.dot(color=col).move_to(bh.get_center() + pos)
            lbl = Text(name, font_size=18, color=col).next_to(d, DOWN, buff=0.1)
            neighbors.add(VGroup(d, lbl))
            edges.add(self.arrow(bh.get_center(), d.get_center(), color=GREY_B))
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.2), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(n) for n in neighbors], lag_ratio=0.2), run_time=0.6)
        gl = Line(UP * 2.5, DOWN * 2.5, color=TEAL, stroke_width=5).shift(RIGHT * 0.5)
        gtxt = Text("range check", font_size=24, color=TEAL).next_to(gl, UP, buff=0.15)
        self.play(Create(gl), Write(gtxt), run_time=0.4)
        p1 = self.arrow(bh.get_center(), n_pos[0] + bh.get_center(), color=WHITE, sw=3, buff=0.15)
        dot1 = self.dot(radius=0.05, color=WHITE).move_to(bh.get_center())
        m1 = Text("Person in range  PASS", font_size=20, color=GREEN).next_to(gl, RIGHT, buff=0.3).shift(UP * 0.8)
        self.play(Create(p1), run_time=0.2)
        self.play(MoveAlongPath(dot1, p1), run_time=0.5)
        self.play(FadeOut(dot1), Write(m1), run_time=0.2)
        p2 = self.arrow(bh.get_center(), n_pos[1] + bh.get_center(), color=WHITE, sw=3, buff=0.15)
        dot2 = self.dot(radius=0.05, color=WHITE).move_to(bh.get_center())
        m2 = Text("Location not in range  BLOCK", font_size=20, color=RED).next_to(gl, RIGHT, buff=0.3).shift(DOWN * 1.0)
        self.play(Create(p2), run_time=0.2)
        self.play(MoveAlongPath(dot2, p2), run_time=0.5)
        self.play(FadeOut(dot2), Write(m2), run_time=0.2)
        self.wait(1.5)
        self.play(FadeOut(rel), FadeOut(gl), FadeOut(gtxt),
                  FadeOut(m1), FadeOut(m2), FadeOut(edges),
                  FadeOut(neighbors), FadeOut(VGroup(bh, bhl)), run_time=0.3)

    def both_gates(self):
        g1 = Line(UP * 2.2, DOWN * 2.2, color=BLUE, stroke_width=4).shift(LEFT * 1.5)
        g1t = Text("Type", font_size=22, color=BLUE).next_to(g1, UP, buff=0.1)
        g2 = Line(UP * 2.2, DOWN * 2.2, color=TEAL, stroke_width=4).shift(RIGHT * 1.5)
        g2t = Text("Range", font_size=22, color=TEAL).next_to(g2, UP, buff=0.1)
        self.play(Create(g1), Write(g1t), Create(g2), Write(g2t), run_time=0.6)
        pts = [LEFT * 4, g1.get_center() + RIGHT * 1.2,
               g2.get_center() + RIGHT * 1.2, RIGHT * 4]
        segs = VGroup()
        for i in range(3):
            segs.add(self.arrow(pts[i], pts[i+1], color=YELLOW, sw=3, buff=0.15))
        glow = self.dot(radius=0.06, color=YELLOW).move_to(pts[0])
        labels = VGroup()
        for txt, col, pos in [("PASS", GREEN, DOWN * 0.5 + LEFT * 0.3),
                               ("PASS", GREEN, DOWN * 0.5 + RIGHT * 1.8),
                               ("ADMITTED", GREEN, pts[3] + RIGHT * 0.6)]:
            l = Text(txt, font_size=20, color=col).move_to(pos)
            labels.add(l)
        for i in range(3):
            self.play(MoveAlongPath(glow, segs[i]), run_time=0.6)
            self.play(Write(labels[i]), run_time=0.15)
        pts2 = [LEFT * 4 + DOWN * 1.5, g1.get_center() + RIGHT * 1.2 + DOWN * 1.5]
        seg2 = self.arrow(pts2[0], pts2[1], color=RED, sw=3, buff=0.15)
        glow2 = self.dot(radius=0.06, color=RED).move_to(pts2[0])
        lbl2 = Text("BLOCKED (wrong type)", font_size=20, color=RED)
        lbl2.move_to(g1.get_center() + RIGHT * 2.5 + DOWN * 1.5)
        self.play(Create(seg2), run_time=0.2)
        self.play(MoveAlongPath(glow2, seg2), run_time=0.5)
        self.play(FadeOut(glow2), Write(lbl2), run_time=0.2)
        props = Text("deterministic  |  conservative  |  O(1)  |  GPU-free",
                     font_size=24, color=GREY_B).to_edge(DOWN, buff=0.4)
        self.play(Write(props), run_time=0.5)
        self.wait(2)


# ──────────────────────────────────────────────────────────
# Scene 3: Trie Construction
# ──────────────────────────────────────────────────────────

class TrieConstruction(PipeScene):
    def construct(self):
        self.show_paths()
        self.build_trie()
        self.query_trie()

    def show_paths(self):
        title = Text("Filtered Paths", font_size=48, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.4)
        paths = [
            "Blue_Hawaii -> film.director -> Norman_Taurog -> nationality -> United_States",
            "Blue_Hawaii -> film.director -> Norman_Taurog -> date_of_birth -> 1901-01-24",
            "Blue_Hawaii -> film.starring -> Elvis_Presley -> nationality -> United_States",
        ]
        colors = [YELLOW, TEAL, ORANGE]
        self.path_labels = VGroup()
        for i, (p, c) in enumerate(zip(paths, colors)):
            lbl = Text(p, font_size=22, color=c)
            lbl.shift(UP * (len(paths) - 1) * 0.5 - i * UP * 0.5)
            self.path_labels.add(lbl)
        self.play(LaggedStart(*[Write(p) for p in self.path_labels], lag_ratio=0.25), run_time=2)
        self.wait(1)
        arrow = Text("---> tokenize", font_size=28, color=GREY_B).next_to(title, DOWN, buff=1.5)
        self.play(Write(arrow), run_time=0.3)
        token_groups = VGroup()
        for i, (p, c) in enumerate(zip(paths, colors)):
            tokens = p.replace(" -> ", " | ")
            tl = Text(tokens, font_size=18, color=c)
            tl.shift(DOWN * 0.5 + self.path_labels[i].get_center() * np.array([0, 1, 0]))
            token_groups.add(tl)
        self.play(LaggedStart(*[TransformFromCopy(self.path_labels[i], token_groups[i])
                                for i in range(3)], lag_ratio=0.2), run_time=1.5)
        self.wait(0.5)
        hint = Text("Each token mapped to an integer ID", font_size=24, color=GREY_B).to_edge(DOWN, buff=0.5)
        self.play(Write(hint), run_time=0.3)
        self.wait(1)
        self.paths_text = token_groups
        self.arrow = arrow
        self.hint = hint

    def build_trie(self):
        self.play(*[FadeOut(p) for p in self.paths_text],
                  FadeOut(self.arrow), FadeOut(self.hint),
                  *[FadeOut(p) for p in self.path_labels], run_time=0.3)
        title = Text("MarisaTrie Construction", font_size=48, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.4)

        trie_nodes = VGroup()
        trie_edges = VGroup()

        def add_node(name, pos, color=WHITE, radius=0.25):
            circle = Circle(radius=radius, color=color, stroke_width=2, fill_opacity=0.1)
            circle.move_to(pos)
            text = Text(name, font_size=16, color=color)
            text.move_to(pos)
            node = VGroup(circle, text)
            trie_nodes.add(node)
            return node

        def add_edge(p1, p2, color=WHITE):
            line = Line(p1, p2, color=color, stroke_width=2)
            trie_edges.add(line)
            return line

        x0, y0 = 0, 2.5
        root = Circle(radius=0.12, color=GREY_D, fill_opacity=0.3)
        root.move_to([x0, y0, 0])
        trie_nodes.add(root)

        n1 = add_node("Blue_Hawaii", [x0, y0 - 1, 0], color=BLUE)
        add_edge(root.get_center(), n1.get_center(), BLUE)
        n2 = add_node("film.director", [x0, y0 - 2.2, 0], color=BLUE)
        add_edge(n1.get_center(), n2.get_center(), BLUE)
        n3 = add_node("Norman_Taurog", [x0, y0 - 3.4, 0], color=YELLOW)
        add_edge(n2.get_center(), n3.get_center(), YELLOW)
        n4l = add_node("nationality", [x0 - 2, y0 - 4.6, 0], color=YELLOW)
        add_edge(n3.get_center(), n4l.get_center(), YELLOW)
        n5l = add_node("United_States", [x0 - 2, y0 - 5.8, 0], color=YELLOW)
        add_edge(n4l.get_center(), n5l.get_center(), YELLOW)
        n4r = add_node("date_of_birth", [x0 + 2, y0 - 4.6, 0], color=TEAL)
        add_edge(n3.get_center(), n4r.get_center(), TEAL)
        n5r = add_node("1901-01-24", [x0 + 2, y0 - 5.8, 0], color=TEAL)
        add_edge(n4r.get_center(), n5r.get_center(), TEAL)

        self.trie_nodes = trie_nodes
        self.trie_edges = trie_edges

        self.play(LaggedStart(*[Create(e) for e in trie_edges], lag_ratio=0.05), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(n, scale=0.5) for n in trie_nodes], lag_ratio=0.05), run_time=1.5)

        shared_highlight = SurroundingRectangle(
            VGroup(n1, n2, n3), color=YELLOW, buff=0.15, stroke_width=2
        )
        shared_lbl = Text("shared prefix", font_size=22, color=YELLOW)
        shared_lbl.next_to(shared_highlight, LEFT, buff=0.3)
        self.play(Create(shared_highlight), Write(shared_lbl), run_time=0.6)
        self.wait(1.5)

        savings = Text("10 token positions  ->  7 stored  (compression!)",
                       font_size=26, color=GREEN).to_edge(DOWN, buff=0.5)
        self.play(Write(savings), run_time=0.5)
        self.wait(1)

        self.trie_data = {
            "root": root, "n1": n1, "n2": n2, "n3": n3,
            "n4l": n4l, "n5l": n5l, "n4r": n4r, "n5r": n5r,
            "shared_hl": shared_highlight, "shared_lbl": shared_lbl,
            "savings": savings,
        }

    def query_trie(self):
        self.play(FadeOut(self.trie_data["shared_hl"]), FadeOut(self.trie_data["shared_lbl"]),
                  FadeOut(self.trie_data["savings"]), run_time=0.3)

        query_title = Text("Trie Query at Decode Time", font_size=40, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(query_title), run_time=0.4)

        dot = self.dot(radius=0.08, color=YELLOW)
        dot.move_to(self.trie_data["root"].get_center())

        path_order = ["root", "n1", "n2", "n3", "n4l", "n5l"]
        node_pos = {k: self.trie_data[k].get_center() for k in path_order}

        segments = VGroup()
        for i in range(len(path_order) - 1):
            seg = Line(node_pos[path_order[i]], node_pos[path_order[i+1]],
                       color=YELLOW, stroke_width=5)
            segments.add(seg)

        step_labels = VGroup()
        step_texts = [
            "generated: Blue_Hawaii", "generated: film.director",
            "generated: Norman_Taurog", "generated: nationality",
            "generated: United_States  (answer!)",
        ]
        for i, txt in enumerate(step_texts):
            lbl = Text(txt, font_size=22, color=YELLOW)
            lbl.shift(RIGHT * 3.5 + UP * (2 - i * 0.4))
            step_labels.add(lbl)

        self.play(FadeIn(dot), run_time=0.2)
        for i in range(5):
            self.play(MoveAlongPath(dot, segments[i]), run_time=0.6)
            self.play(Write(step_labels[i]), run_time=0.15)
        self.wait(0.5)

        constraint = Text("valid next tokens = children of current node",
                          font_size=28, color=GREEN).to_edge(DOWN, buff=0.5)
        self.play(Write(constraint), run_time=0.4)
        self.wait(1.5)
        self.play(FadeOut(query_title), FadeOut(constraint), run_time=0.3)

        children_title = Text("At Norman_Taurog: 2 valid next tokens", font_size=32, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(children_title), run_time=0.4)

        n3 = self.trie_data["n3"]
        n4l = self.trie_data["n4l"]
        n4r = self.trie_data["n4r"]
        e_l = Line(n3.get_center(), n4l.get_center(), color=YELLOW, stroke_width=6)
        e_r = Line(n3.get_center(), n4r.get_center(), color=TEAL, stroke_width=6)

        children_list = VGroup(
            Text("1. nationality   -->   United_States", font_size=24, color=YELLOW),
            Text("2. date_of_birth -->   1901-01-24", font_size=24, color=TEAL),
        )
        children_list.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        children_list.next_to(n3, RIGHT, buff=1.5)

        self.play(Transform(self.trie_edges[-4], e_l), Transform(self.trie_edges[-2], e_r),
                  Write(children_list), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(children_title), FadeOut(children_list),
                  FadeOut(dot), FadeOut(step_labels),
                  *[FadeOut(s) for s in segments], run_time=0.3)


# ──────────────────────────────────────────────────────────
# Scene 4: Constrained Decoding V1
# ──────────────────────────────────────────────────────────

class ConstrainedDecodingV1(PipeScene):
    def construct(self):
        self.setup_llm()
        self.free_generation()
        self.enter_path_mode()
        self.constrained_step("Blue_Hawaii", "  Blue_Hawaii", 0)
        self.constrained_step("->", "  ->", 1)
        self.constrained_step("film.director", "  film.director", 2)
        self.constrained_step("Norman_Taurog", "  Norman_Taurog", 3)
        self.exit_path_mode()
        self.summary()

    def setup_llm(self):
        # Output text box on the left
        box = Rectangle(width=7, height=4, color=GREY_D, stroke_width=2)
        box.shift(LEFT * 2.5)
        box_lbl = Text("LLM Output", font_size=22, color=GREY_B).next_to(box, UP, buff=0.15)
        self.play(Create(box), Write(box_lbl), run_time=0.5)

        # Question prefix already shown
        prefix = Text('Q: What is the nationality\n   of the director of\n   Blue Hawaii?\n\nA:', font_size=22, color=GREY_B)
        prefix.move_to(box.get_center() + UP * 0.5 + LEFT * 2.5)
        prefix.align_to(box, LEFT)
        prefix.shift(RIGHT * 0.3)
        self.play(Write(prefix), run_time=0.6)

        # Trie on the right (simplified)
        trie_title = Text("Trie", font_size=22, color=GREY_B)
        trie_title.shift(RIGHT * 3.5 + UP * 3)
        self.trie_nodes = VGroup()
        self.trie_edges = VGroup()

        def tnode(name, pos, color=GREY_B, r=0.2):
            c = Circle(radius=r, color=color, stroke_width=2, fill_opacity=0.08)
            c.move_to(pos)
            t = Text(name, font_size=12, color=color)
            t.move_to(pos)
            n = VGroup(c, t)
            self.trie_nodes.add(n)
            return n

        def tedge(p1, p2, color=GREY_B):
            l = Line(p1, p2, color=color, stroke_width=1.5)
            self.trie_edges.add(l)
            return l

        x, y = 3.5, 2.0
        r = tnode("root", [x, y, 0], GREY_D, 0.08)
        n_bh = tnode("Blue_Hawaii", [x, y - 0.7, 0], BLUE)
        tedge(r.get_center(), n_bh.get_center(), BLUE)
        n_fd = tnode("film.director", [x, y - 1.3, 0], BLUE)
        tedge(n_bh.get_center(), n_fd.get_center(), BLUE)
        n_nt = tnode("Norman_Taurog", [x, y - 1.9, 0], GREEN)
        tedge(n_fd.get_center(), n_nt.get_center(), GREEN)
        n_nat = tnode("nationality", [x - 0.8, y - 2.5, 0], YELLOW)
        tedge(n_nt.get_center(), n_nat.get_center(), YELLOW)
        n_us = tnode("United_States", [x - 0.8, y - 3.1, 0], YELLOW)
        tedge(n_nat.get_center(), n_us.get_center(), YELLOW)
        n_dob = tnode("date_of_birth", [x + 0.8, y - 2.5, 0], TEAL)
        tedge(n_nt.get_center(), n_dob.get_center(), TEAL)
        n_dt = tnode("1901-01-24", [x + 0.8, y - 3.1, 0], TEAL)
        tedge(n_dob.get_center(), n_dt.get_center(), TEAL)

        self.play(Write(trie_title), run_time=0.2)
        self.play(LaggedStart(*[FadeIn(n, scale=0.5) for n in self.trie_nodes], lag_ratio=0.03), run_time=0.8)
        self.play(LaggedStart(*[Create(e) for e in self.trie_edges], lag_ratio=0.03), run_time=0.8)

        # Cursor dot for trie
        self.trie_cursor = self.dot(radius=0.06, color=YELLOW)
        self.trie_cursor.move_to(r.get_center())
        self.play(FadeIn(self.trie_cursor), run_time=0.2)

        self.output_box = box
        self.prefix = prefix
        self.output_text = VGroup()

        # Bar chart placeholder (probabilities)
        self.bars = VGroup()
        self.bar_labels = VGroup()
        self.bar_title = Text("Token Probabilities", font_size=18, color=GREY_B)
        self.bar_title.move_to(RIGHT * 3.5 + DOWN * 2.5)

    def make_bars(self, tokens, probs, highlight=None):
        self.play(FadeOut(self.bars), FadeOut(self.bar_labels), run_time=0.1)
        self.bars = VGroup()
        self.bar_labels = VGroup()
        for i, (tok, prob) in enumerate(zip(tokens, probs)):
            w = prob * 3.5
            col = highlight if i == highlight else GREY_B
            bar = Rectangle(width=w, height=0.2, color=col, fill_opacity=0.6, stroke_width=1)
            bar.move_to(RIGHT * 2.5 + DOWN * (1.5 - i * 0.25))
            bar.align_to(ORIGIN + RIGHT * 2.5, LEFT)
            lbl = Text(tok, font_size=12, color=col)
            lbl.next_to(bar, RIGHT, buff=0.1)
            self.bars.add(bar)
            self.bar_labels.add(lbl)

        self.play(FadeIn(self.bar_title), run_time=0.15)
        self.play(LaggedStart(*[Create(b) for b in self.bars], lag_ratio=0.05), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in self.bar_labels], lag_ratio=0.05), run_time=0.3)
        self.wait(0.3)

    def add_output(self, text, color=WHITE):
        t = Text(text, font_size=22, color=color)
        t.move_to(self.output_box.get_center() + DOWN * 1.5 + LEFT * 2.5)
        t.align_to(self.output_box, LEFT)
        t.shift(RIGHT * 0.3)
        self.play(Write(t), run_time=0.3)
        self.output_text.add(t)
        return t

    def free_generation(self):
        # Show LLM generating initial text freely
        t1 = self.add_output("The answer is ")
        self.wait(0.3)

    def enter_path_mode(self):
        path_tag = self.add_output("<PATH>", color=YELLOW)
        self.wait(0.2)

        # Glow effect on trie
        glow = SurroundingRectangle(self.trie_nodes, color=YELLOW, buff=0.1, stroke_width=2)
        glow.set_fill(YELLOW, opacity=0.03)
        label = Text("Trie active", font_size=20, color=YELLOW)
        label.next_to(glow, UP, buff=0.1)
        self.play(Create(glow), Write(label), run_time=0.4)

        # Move cursor to first child
        self.play(self.trie_cursor.animate.move_to(
            self.trie_nodes[1].get_center()),
            run_time=0.4)

        self.trie_glow = glow
        self.trie_glow_label = label

    def constrained_step(self, token, output_str, idx):
        # Show probability bars before masking
        tokens = ["Blue_Hawaii", "Elvis_Presley", "Hawaii", "Norman_Taurog"]
        probs = [0.35, 0.30, 0.20, 0.15]
        if idx == 2:  # relation step
            tokens = ["film.director", "film.starring", "film.location"]
            probs = [0.50, 0.30, 0.20]
        elif idx == 3:  # entity step
            tokens = ["Norman_Taurog", "United_States", "Date_1901"]
            probs = [0.60, 0.25, 0.15]

        # Show unmasked
        self.make_bars(tokens, probs)

        # "Masking" effect: highlight the chosen token, dim others
        highlight_idx = tokens.index(token) if token in tokens else 0
        self.make_bars(tokens, probs, highlight=YELLOW)

        # Add output
        self.add_output(output_str, color=YELLOW)

        # Advance trie cursor
        next_node_idx = min(idx + 2, len(self.trie_nodes) - 1)
        if idx == 0:
            target = self.trie_nodes[2].get_center()
        elif idx == 1:
            target = self.trie_nodes[3].get_center()
        elif idx == 2:
            target = self.trie_nodes[4].get_center()
        else:
            target = self.trie_nodes[5].get_center()
        self.play(self.trie_cursor.animate.move_to(target), run_time=0.3)
        self.wait(0.3)

    def exit_path_mode(self):
        self.play(FadeOut(self.bars), FadeOut(self.bar_labels),
                  FadeOut(self.bar_title), run_time=0.2)
        self.add_output("</PATH>", color=YELLOW)
        self.wait(0.2)

    def summary(self):
        self.play(FadeOut(self.trie_glow), FadeOut(self.trie_glow_label), run_time=0.3)

        # Guarantee text
        summary_box = SurroundingRectangle(self.output_text, color=GREEN, buff=0.2)
        guarantee = Text("Zero structural hallucination guaranteed",
                         font_size=28, color=GREEN)
        guarantee.to_edge(DOWN, buff=0.5)

        self.play(Create(summary_box), Write(guarantee), run_time=0.6)

        # Mechanism annotation
        mech = VGroup(
            Text("prefix_allowed_tokens_fn:", font_size=24, color=GREY_B),
            Text("trie.get(prefix) --> valid next tokens", font_size=24, color=TEAL),
            Text("logits[invalid] = -inf   (masked)", font_size=24, color=RED),
        )
        mech.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        mech.next_to(summary_box, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(m, shift=UP) for m in mech], lag_ratio=0.2), run_time=1)
        self.wait(2)
