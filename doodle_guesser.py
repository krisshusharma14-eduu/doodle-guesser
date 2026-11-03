import json, os, random, time, tkinter as tk
from tkinter import ttk
import numpy as np, joblib
from PIL import Image, ImageDraw
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from common import preprocess, decide, LETTER_WORDS, LETTER_EMOJI, EMOJI, pic
from shapes import SHAPES, trace_score

SIZE, BRUSH, HISTORY = 420, 16, "history.json"
BG, PURPLE, PINK, BLUE, GREEN, ORANGE, INK = "#FFF4D6", "#7C4DFF", "#FF6FA5", "#29B6F6", "#43C463", "#FFA726", "#33345A"
FONT = "Comic Sans MS"
COLORS = ["#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5", "#8e24aa", "#6d4c41", "#212121"]


class Btn(tk.Label):
    """Colourful button made from a Label (normal buttons ignore colours on macOS)."""
    def __init__(self, parent, text, cmd, bg, size=14, **kw):
        super().__init__(parent, text=text, bg=bg, fg="white", font=(FONT, size, "bold"), padx=14, pady=8,
                         cursor="hand2", highlightthickness=0, highlightbackground=INK, **kw)
        self.bind("<Button-1>", lambda e: cmd())

    def select(self, on):
        self.config(highlightthickness=4 if on else 0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Doodle Guesser")
        self.geometry("1100x760")
        self.configure(bg=BG)
        self.models = {m: joblib.load(f"models/{m}.joblib")
                       for m in ("objects", "letters") if os.path.exists(f"models/{m}.joblib")}
        if not self.models:
            raise SystemExit("Run: python train_models.py first")
        self.metrics = json.load(open("models/metrics.json")) if os.path.exists("models/metrics.json") else {}
        self.mode, self.challenge, self.details, self.erase = "objects", False, False, False
        self.color, self.shape = COLORS[4], None
        self.history = self.load_history()
        self.live_job = self.commit_job = self.pending = self.last = self.target = None
        self.new_image()

        bar = tk.Frame(self, bg=PURPLE); bar.pack(fill="x")
        tk.Label(bar, text=pic("🎨 ") + "Doodle Guesser", font=(FONT, 28, "bold"), bg=PURPLE,
                 fg="white").pack(side="left", padx=20, pady=10)
        self.nav = Btn(bar, pic("📊 ") + "Dashboard", self.toggle_view, ORANGE)
        self.nav.pack(side="right", padx=15)
        self.play, self.dash = tk.Frame(self, bg=BG), tk.Frame(self, bg=BG)
        self.view = "play"
        self.build_play(); self.build_dash()
        self.play.pack(fill="both", expand=True)
        self.set_mode("objects" if "objects" in self.models else "letters")

    # ---------- data ----------
    def load_history(self):
        try:
            return json.load(open(HISTORY))
        except Exception:
            return []

    def save_history(self):
        try:
            json.dump(self.history, open(HISTORY, "w"))
        except Exception:
            pass

    def new_image(self):
        self.img = Image.new("L", (SIZE, SIZE), 0)      # hidden image the AI looks at (white on black)
        self.draw = ImageDraw.Draw(self.img)

    # ---------- UI ----------
    def build_play(self):
        left = tk.Frame(self.play, bg=BG); left.pack(side="left", padx=20, pady=12)
        self.canvas = tk.Canvas(left, width=SIZE, height=SIZE, bg="white", cursor="pencil",
                                highlightthickness=6, highlightbackground=PINK)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        row = tk.Frame(left, bg=BG); row.pack(pady=10)
        self.swatches = {}
        for c in COLORS:
            sw = tk.Label(row, bg=c, width=3, height=1, cursor="hand2", highlightbackground=INK)
            sw.pack(side="left", padx=3)
            sw.bind("<Button-1>", lambda e, c=c: self.pick(c))
            self.swatches[c] = sw
        row2 = tk.Frame(left, bg=BG); row2.pack()
        self.erase_btn = Btn(row2, pic("🧽 ") + "Eraser", self.toggle_erase, BLUE, size=12)
        self.erase_btn.pack(side="left", padx=5)
        Btn(row2, pic("🗑️ ") + "Clear", self.clear, PINK, size=12).pack(side="left", padx=5)
        self.pick(self.color)

        right = tk.Frame(self.play, bg=BG); right.pack(side="left", fill="both", expand=True, padx=15, pady=12)
        m = tk.Frame(right, bg=BG); m.pack(anchor="w")
        self.mode_btns = {}
        for key, text, col in [("objects", pic("🐱 ") + "Objects", BLUE), ("letters", pic("🔤 ") + "Letters A-Z", PINK),
                               ("trace", pic("✏️ ") + "Trace & Practise", GREEN)]:
            b = Btn(m, text, lambda k=key: self.set_mode(k), col, size=13)
            b.pack(side="left", padx=4); self.mode_btns[key] = b
        self.prompt = tk.Label(right, text="", font=(FONT, 17, "bold"), fg=PURPLE, bg=BG,
                               wraplength=500, justify="left")
        self.prompt.pack(anchor="w", pady=(12, 0))
        card = tk.Frame(right, bg="white", highlightthickness=4, highlightbackground=BLUE)
        card.pack(fill="x", pady=10)
        self.big = tk.Label(card, text=pic("✏️"), font=(FONT, 70), bg="white"); self.big.pack(pady=(8, 0))
        self.result = tk.Label(card, text="Start drawing!", font=(FONT, 20, "bold"), bg="white", fg="gray",
                               wraplength=480, justify="center")
        self.result.pack(padx=10)
        self.stars = tk.Label(card, text="", font=(FONT, 30), bg="white", fg=ORANGE); self.stars.pack(pady=(0, 8))
        r = tk.Frame(right, bg=BG); r.pack(anchor="w")
        self.next_btn = Btn(r, "Next " + pic("➡️"), self.next_challenge, GREEN, size=13)
        self.next_btn.pack(side="left", padx=(0, 6))
        self.chal_btn = Btn(r, pic("🎯 ") + "Challenge: OFF", self.toggle_challenge, PURPLE, size=13)
        self.chal_btn.pack(side="left")
        self.det_btn = Btn(right, pic("🔍 ") + "AI details (for teachers)", self.toggle_details, INK, size=11)
        self.det_btn.pack(anchor="w", pady=8)
        self.det_frame = tk.Frame(right, bg=BG)
        self.bars = []
        for _ in range(3):
            lab = tk.Label(self.det_frame, text="", bg=BG, fg=INK, font=(FONT, 11))
            bar = ttk.Progressbar(self.det_frame, length=300, maximum=100)
            lab.pack(anchor="w"); bar.pack(anchor="w", pady=(0, 4))
            self.bars.append((lab, bar))
        self.known = tk.Label(right, fg=INK, bg=BG, font=(FONT, 12), wraplength=520, justify="left")
        self.known.pack(side="bottom", anchor="w")

    def build_dash(self):
        self.summary = tk.Label(self.dash, font=(FONT, 14), justify="left", bg=BG, fg=INK)
        self.summary.pack(anchor="w", padx=15, pady=10)
        self.fig = Figure(figsize=(9.5, 3.8), dpi=100, facecolor=BG)
        self.fcanvas = FigureCanvasTkAgg(self.fig, master=self.dash)
        self.fcanvas.get_tk_widget().pack(fill="both", expand=True, padx=10)
        Btn(self.dash, "Refresh", self.refresh_dashboard, BLUE, size=12).pack(pady=8)

    def toggle_view(self):
        self.commit()
        if self.view == "play":
            self.play.pack_forget(); self.dash.pack(fill="both", expand=True)
            self.view = "dash"; self.nav.config(text=pic("🎮 ") + "Play"); self.refresh_dashboard()
        else:
            self.dash.pack_forget(); self.play.pack(fill="both", expand=True)
            self.view = "play"; self.nav.config(text=pic("📊 ") + "Dashboard")

    def update_known(self):
        if self.mode == "trace":
            self.known.config(text="Shapes to trace: " + ", ".join(SHAPES))
        elif self.mode == "letters":
            self.known.config(text="I know: " + "  ".join(f"{k}{pic(LETTER_EMOJI[k])}" for k in LETTER_WORDS))
        else:
            cls = [c for c in self.models["objects"].classes_ if c != "other"]
            self.known.config(text="I know these: " + "   ".join(f"{pic(EMOJI.get(c, ''))} {c}" for c in cls))

    def toggle_details(self):
        self.details = not self.details
        if self.details:
            self.det_frame.pack(anchor="w", pady=4, after=self.det_btn)
        else:
            self.det_frame.pack_forget()

    def toggle_challenge(self):
        self.challenge = not self.challenge
        self.chal_btn.config(text=pic("🎯 ") + f"Challenge: {'ON' if self.challenge else 'OFF'}")
        self.next_challenge()

    def pick(self, c):
        self.color, self.erase = c, False
        for k, sw in self.swatches.items():
            sw.config(highlightthickness=3 if k == c else 0)
        if hasattr(self, "erase_btn"):
            self.erase_btn.select(False)

    def toggle_erase(self):
        self.erase = not self.erase
        self.erase_btn.select(self.erase)

    # ---------- drawing ----------
    def on_down(self, e):
        self.cancel(); self.last = (e.x, e.y)
        w = BRUSH * 2 if self.erase else BRUSH
        r, col = w / 2, "white" if self.erase else self.color
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill=col, outline=col)
        self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=0 if self.erase else 255)

    def on_move(self, e):
        if not self.last:
            return
        (lx, ly) = self.last
        w = BRUSH * 2 if self.erase else BRUSH
        r, col, fill = w / 2, "white" if self.erase else self.color, 0 if self.erase else 255
        self.canvas.create_line(lx, ly, e.x, e.y, fill=col, width=w, capstyle="round")
        self.draw.line([lx, ly, e.x, e.y], fill=fill, width=int(w))
        self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=fill)
        self.last = (e.x, e.y)
        if self.live_job is None and self.mode != "trace":      # throttled live prediction
            self.live_job = self.after(200, self.predict)

    def on_up(self, e):
        self.last = None
        self.cancel()
        self.live_job = self.after(500 if self.mode == "trace" else 30, self.evaluate)
        self.commit_job = self.after(1500, self.commit)          # log the attempt after a pause

    def cancel(self):
        for j in ("live_job", "commit_job"):
            if getattr(self, j):
                self.after_cancel(getattr(self, j)); setattr(self, j, None)

    # ---------- prediction ----------
    def evaluate(self):
        self.check_trace() if self.mode == "trace" else self.predict()

    def predict(self):
        self.live_job = None
        x = preprocess(self.img)
        if x is None:
            self.show("Keep drawing...", "gray", "✏️"); self.pending = None
            return
        mode = self.mode
        model = self.models[mode]
        label, conf, top = decide(model.classes_, model.predict_proba([x])[0])
        ok = label is not None and (not self.target or label == self.target)
        self.pending = dict(mode=mode, guess=label, conf=conf, target=self.target, ok=ok)
        for (lab, bar), t in zip(self.bars, top):
            lab.config(text=f"{t[0]}  {t[1] * 100:.0f}%"); bar["value"] = t[1] * 100
        good = "#2e7d32"
        if label is None:
            self.show("Attempt failed - I couldn't recognise that. Try again!", "#c62828", "🤔")
            return
        if mode == "letters":
            txt = f"Letter {label} - {label} for {LETTER_WORDS[label]}!"
            emoji = LETTER_EMOJI[label]
        else:
            txt = f"I think it is a {label}!"
            emoji = EMOJI.get(label, "🎨")
        if self.target and ok:
            txt = "Correct! " + txt
        elif self.target:
            txt += f"  (I need: {self.target})"
        self.show(txt, good, emoji)
        if self.target and ok:
            self.party()

    def check_trace(self):
        self.live_job = None
        ink = np.array(self.img) > 0
        if ink.sum() < 50 or not self.shape:
            self.show("Trace the dotted line!", "gray", "✏️"); self.pending = None
            return
        score = trace_score(ink, SHAPES[self.shape], SIZE)
        pct = round(score * 100)
        n = 3 if pct >= 85 else 2 if pct >= 70 else 1 if pct >= 55 else 0
        ok = pct >= 55
        self.pending = dict(mode="trace", guess=self.shape if ok else None, conf=score, target=self.shape, ok=ok)
        msg = ["Keep going along the dots!", "Good try! Go slowly along the dots.", "Great job!", "Amazing tracing!"][n]
        emoji = ["💪", "🙂", "😀", "🏆"][n]
        self.show(f"{msg} {pct}%", "#2e7d32" if ok else "#c62828", emoji, "★" * n + "☆" * (3 - n))
        if n == 3:
            self.party()

    def show(self, text, color, emoji="✏️", stars=""):
        self.result.config(text=text, fg=color)
        self.big.config(text=pic(emoji))
        self.stars.config(text=stars)

    def party(self):
        self.canvas.delete("party")
        self.canvas.create_text(SIZE / 2, SIZE / 2, text=pic("🎉") or "WOW!", font=(FONT, 90), tags="party")
        self.after(1300, lambda: self.canvas.delete("party"))

    def commit(self):
        if self.live_job:
            self.after_cancel(self.live_job); self.live_job = None; self.evaluate()
        if self.commit_job:
            self.after_cancel(self.commit_job); self.commit_job = None
        if self.pending:
            self.pending["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.history.append(self.pending); self.pending = None
            self.save_history()

    # ---------- controls ----------
    def clear(self):
        self.commit()
        self.canvas.delete("all"); self.new_image()
        if self.mode == "trace" and self.shape:
            self.draw_guide()
        self.show("Start drawing!" if self.mode != "trace" else "Trace the dotted line!", "gray", "✏️")
        for lab, bar in self.bars:
            lab.config(text=""); bar["value"] = 0

    def draw_guide(self):
        for stroke in SHAPES[self.shape]:
            pts = [c * SIZE for p in stroke for c in p]
            self.canvas.create_line(*pts, fill="#90A4AE", width=6, dash=(2, 10), capstyle="round", tags="guide")
        x, y = [v * SIZE for v in SHAPES[self.shape][0][0]]
        self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill=GREEN, outline="white", width=3, tags="guide")
        self.canvas.create_text(x, y - 26, text="START", font=(FONT, 11, "bold"), fill=GREEN, tags="guide")

    def next_challenge(self):
        if self.mode == "trace":
            self.shape = random.choice([s for s in SHAPES if s != self.shape])
            self.target = self.shape
            self.clear()
            self.prompt.config(text=f"{pic(EMOJI.get(self.shape, ''))}Trace the dotted {self.shape}! Start at the green dot.")
        elif self.challenge:
            self.shape = None
            self.clear()
            cls = [c for c in self.models[self.mode].classes_ if c != "other"]
            self.target = random.choice(cls)
            self.prompt.config(text=f"Challenge: draw {'the letter ' if self.mode == 'letters' else 'a '}{self.target}"
                                    f" {pic(EMOJI.get(self.target, '')) if self.mode == 'objects' else ''}")
        else:
            self.shape = self.target = None
            self.clear()
            self.prompt.config(text="Draw anything you like!")

    def set_mode(self, m):
        if m in ("objects", "letters") and m not in self.models:
            return
        self.commit()
        self.mode = m
        for k, b in self.mode_btns.items():
            b.select(k == m)
        if m == "trace":
            self.chal_btn.pack_forget()
        elif not self.chal_btn.winfo_ismapped():
            self.chal_btn.pack(side="left")
        self.update_known(); self.next_challenge()

    # ---------- dashboard ----------
    def refresh_dashboard(self):
        h = [r for r in self.history if r.get("mode") != "trace"]
        tr = [r for r in self.history if r.get("mode") == "trace"]
        total = len(h)
        rec = sum(1 for r in h if r["guess"])
        ch = [r for r in h if r["target"]]
        ch_ok = sum(1 for r in ch if r["ok"])
        acc = ", ".join(f"{k} {v * 100:.1f}%" for k, v in self.metrics.items()) or "n/a"
        tr_avg = sum(r["conf"] for r in tr) / len(tr) * 100 if tr else 0
        self.summary.config(text=(
            f"Attempts: {total}    Recognised: {rec} ({rec / total * 100 if total else 0:.0f}%)    "
            f"Failed: {total - rec}\nChallenge accuracy: {ch_ok}/{len(ch)}    Model test accuracy: {acc}\n"
            f"Tracing practice: {len(tr)} tries, average score {tr_avg:.0f}%"))
        self.fig.clear()
        a1, a2, a3 = self.fig.subplots(1, 3)
        counts = {}
        for r in h:
            if r["guess"]:
                counts[r["guess"]] = counts.get(r["guess"], 0) + 1
        items = sorted(counts.items(), key=lambda kv: kv[1])[-8:]
        if items:
            a1.barh([k for k, _ in items], [v for _, v in items], color="#42a5f5")
        a1.set_title("Most guessed")
        a2.bar(["Recognised", "Failed"], [rec, total - rec], color=["#66bb6a", "#ef5350"])
        a2.set_title("Outcomes")
        last = h[-50:]
        a3.plot(range(len(last)), [r["conf"] * 100 for r in last], marker="o", color="#7e57c2")
        a3.set_ylim(0, 100); a3.set_title("Confidence % (last 50)")
        self.fig.tight_layout()
        self.fcanvas.draw()


if __name__ == "__main__":
    App().mainloop()