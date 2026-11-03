"""
Doodle Guesser - AI/ML mini project
Draw a doodle and the AI guesses what it is.

Algorithms: KNN, Decision Tree, ANN (MLP neural network), K-Means (+ PCA)
Dataset   : Google Quick, Draw! (28x28 doodles, 10 categories)

Usage:
    python doodle_guesser.py download     # step 1: get the dataset (internet needed)
    python doodle_guesser.py train        # step 2: train models, make graphs
    python doodle_guesser.py app          # step 3: draw and let the AI guess
"""
import argparse
import os
import random
import sys
import time
import urllib.request
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

CATEGORIES = ["apple", "cat", "house", "car", "tree",
              "sun", "fish", "star", "umbrella", "clock"]
SCRIBBLE = "scribble"                      # 11th class: "this is not one of my doodles"
ALL_CLASSES = CATEGORIES + [SCRIBBLE]
DATA_DIR, OUT_DIR, MODEL_FILE = "data", "results", "doodle_models.joblib"
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

# "I'm not sure" rules (tune these if the app is too strict or too relaxed)
CONFIDENCE_MIN = 0.55   # the 3 models together must be at least this sure
MIN_AGREE = 2           # at least 2 of the 3 models must pick the same object
MIN_DRAW_SIZE = 50      # drawing must be at least 50 pixels wide or tall (canvas is 300)


# ----------------------------------------------------------------------------
# 1. DOWNLOAD (only the first N doodles of each category, a few MB each)
# ----------------------------------------------------------------------------
def download(n):
    os.makedirs(DATA_DIR, exist_ok=True)
    for cat in CATEGORIES:
        path = os.path.join(DATA_DIR, f"{cat}.npy")
        if os.path.exists(path):
            print(f"{cat:9s} already downloaded")
            continue
        need = 256 + n * 784
        req = urllib.request.Request(URL.format(cat), headers={"Range": f"bytes=0-{need - 1}"})
        raw = b""
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                while len(raw) < need:
                    chunk = r.read(need - len(raw))
                    if not chunk:
                        break
                    raw += chunk
        except Exception as e:
            sys.exit(f"Download failed for '{cat}': {e}\nCheck your internet connection and run again.")
        # skip the .npy header to get to the pixel data
        hlen = int.from_bytes(raw[8:10], "little") if raw[6] == 1 else int.from_bytes(raw[8:12], "little")
        offset = (10 if raw[6] == 1 else 12) + hlen
        arr = np.frombuffer(raw[offset:], dtype=np.uint8)
        arr = arr[: len(arr) // 784 * 784].reshape(-1, 784)[:n]
        np.save(path, arr)
        print(f"{cat:9s} downloaded {len(arr)} doodles")
    print("Done. Now run:  python doodle_guesser.py train")


# ----------------------------------------------------------------------------
# 2. PRE-PROCESSING (same function for training images AND your drawings)
# ----------------------------------------------------------------------------
def normalize(img):
    """Crop to the drawing, make it square, centre it in a 28x28 frame.
    This makes the model ignore WHERE you drew and HOW BIG you drew."""
    ys, xs = np.where(img > 40)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    sq[top:top + h, left:left + w] = crop
    small = np.array(Image.fromarray(sq).resize((20, 20), Image.LANCZOS))
    out = np.zeros((28, 28), np.uint8)
    out[4:24, 4:24] = small
    return out


def load_data(n):
    X, y = [], []
    for i, cat in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, f"{cat}.npy")
        if not os.path.exists(path):
            sys.exit("Dataset not found. Run:  python doodle_guesser.py download")
        a = np.load(path)[:n]
        X.append(np.array([normalize(r.reshape(28, 28)).ravel() for r in a], np.float32) / 255.0)
        y.append(np.full(len(a), i))
    return np.vstack(X), np.concatenate(y)


def make_scribbles(n, seed=0):
    """Make random scribbles (lines, zigzags, blobs) drawn exactly like the app draws:
    300x300 canvas, thick round brush. They teach the models to say 'not a doodle'."""
    from PIL import ImageDraw
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        im = Image.new("L", (300, 300), 0)
        d = ImageDraw.Draw(im)
        width = rnd.randint(10, 18)
        kind = rnd.choice(["walk", "zigzag", "lines", "blobs", "spiral"])
        if kind == "walk":                                   # wandering line
            x, y = rnd.randint(40, 260), rnd.randint(40, 260)
            for _ in range(rnd.randint(4, 16)):
                nx = min(280, max(20, x + rnd.randint(-110, 110)))
                ny = min(280, max(20, y + rnd.randint(-110, 110)))
                d.line([x, y, nx, ny], fill=255, width=width)
                x, y = nx, ny
        elif kind == "zigzag":                               # back-and-forth scribble
            y = rnd.randint(30, 120)
            for i in range(rnd.randint(6, 20)):
                x0, x1 = rnd.randint(20, 120), rnd.randint(160, 280)
                d.line([x0, y, x1, y + rnd.randint(5, 25)], fill=255, width=width)
                y += rnd.randint(8, 22)
        elif kind == "lines":                                # a few straight lines
            for _ in range(rnd.randint(1, 4)):
                d.line([rnd.randint(20, 280), rnd.randint(20, 280),
                        rnd.randint(20, 280), rnd.randint(20, 280)], fill=255, width=width)
        elif kind == "blobs":                                # overlapping blobs / filled shapes
            for _ in range(rnd.randint(1, 5)):
                x, y = rnd.randint(30, 200), rnd.randint(30, 200)
                w, h = rnd.randint(30, 100), rnd.randint(30, 100)
                if rnd.random() < 0.5:
                    d.ellipse([x, y, x + w, y + h], fill=255)
                else:
                    d.rectangle([x, y, x + w, y + h], fill=255)
        else:                                                # messy spiral
            cx, cy = rnd.randint(100, 200), rnd.randint(100, 200)
            pts = [(cx + np.cos(t) * t * rnd.uniform(5, 9), cy + np.sin(t) * t * rnd.uniform(5, 9))
                   for t in np.linspace(0, rnd.uniform(8, 18), 60)]
            d.line(pts, fill=255, width=width)
        arr = np.array(im)
        ys, xs = np.where(arr > 40)
        if len(xs) == 0 or max(ys.max() - ys.min(), xs.max() - xs.min()) < MIN_DRAW_SIZE:
            arr[100:200, 100:200] = 255                      # make sure it is not "tiny"
        out.append(normalize(arr).ravel())
    return np.array(out, np.float32) / 255.0


# ----------------------------------------------------------------------------
# 3. TRAIN + EVALUATE + GRAPHS
# ----------------------------------------------------------------------------
def train(args):
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, adjusted_rand_score,
                                 f1_score, homogeneity_score)
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)
    print("Loading and pre-processing doodles ...")
    X, y = load_data(args.samples)
    classes = CATEGORIES
    if args.scribbles > 0:
        print(f"Making {args.scribbles} random scribbles (the 'not a doodle' class) ...")
        Xs = make_scribbles(args.scribbles)
        X = np.vstack([X, Xs])
        y = np.concatenate([y, np.full(len(Xs), len(CATEGORIES))])
        classes = ALL_CLASSES
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Total {len(X)} images | train {len(Xtr)} | test {len(Xte)} | classes {len(classes)}")

    # --- sample grid -------------------------------------------------------
    fig, ax = plt.subplots(len(classes), 10, figsize=(10, len(classes)))
    for c in range(len(classes)):
        idx = np.where(y == c)[0][:10]
        for j in range(10):
            ax[c, j].imshow(X[idx[j]].reshape(28, 28), cmap="gray_r")
            ax[c, j].axis("off")
        ax[c, 0].set_title(classes[c], fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/1_dataset_samples.png", dpi=130)
    plt.close()

    # --- PCA (784 pixels -> 50 features) --------------------------------
    pca = PCA(n_components=args.pca, random_state=42).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    print(f"PCA: 784 features -> {args.pca} features "
          f"({pca.explained_variance_ratio_.sum() * 100:.1f}% of information kept)")

    # --- the three supervised models ---------------------------------------
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80, early_stopping=True,
                             random_state=42),
    }
    rows, preds = [], {}
    for name, m in models.items():
        t = time.time(); m.fit(Ptr, ytr); fit_t = time.time() - t
        t = time.time(); p = m.predict(Pte); pred_t = time.time() - t
        preds[name] = p
        rows.append({"Model": name,
                     "Train accuracy %": round(accuracy_score(ytr, m.predict(Ptr)) * 100, 2),
                     "Test accuracy %": round(accuracy_score(yte, p) * 100, 2),
                     "Test F1 (macro) %": round(f1_score(yte, p, average="macro") * 100, 2),
                     "Training time (s)": round(fit_t, 2),
                     "Prediction time (s)": round(pred_t, 2)})
        print(f"{name:14s} test accuracy {rows[-1]['Test accuracy %']:.2f}%")
    table = pd.DataFrame(rows)
    table.to_csv(f"{OUT_DIR}/results_table.csv", index=False)
    print("\n" + table.to_string(index=False) + "\n")

    # --- comparison bar chart ----------------------------------------------
    plt.figure(figsize=(7, 4))
    xs = np.arange(len(table))
    plt.bar(xs - 0.2, table["Train accuracy %"], 0.4, label="Train")
    plt.bar(xs + 0.2, table["Test accuracy %"], 0.4, label="Test")
    plt.xticks(xs, table["Model"])
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Model comparison")
    for i, v in enumerate(table["Test accuracy %"]):
        plt.text(i + 0.2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/2_model_comparison.png", dpi=150)
    plt.close()

    # --- KNN: effect of K --------------------------------------------------
    ks = [1, 3, 5, 7, 9, 11, 15]
    k_acc = [accuracy_score(yte, KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                            .fit(Ptr, ytr).predict(Pte)) * 100 for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, k_acc, "o-")
    plt.xlabel("K (number of neighbours)")
    plt.ylabel("Test accuracy (%)")
    plt.title("KNN: choosing K")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3_knn_k_sweep.png", dpi=150)
    plt.close()
    print("KNN best K:", ks[int(np.argmax(k_acc))])

    # --- Decision tree: effect of depth (overfitting) -------------------
    depths = [2, 4, 6, 8, 10, 15, 20, 30]
    tr_a, te_a = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Ptr, ytr)
        tr_a.append(accuracy_score(ytr, t.predict(Ptr)) * 100)
        te_a.append(accuracy_score(yte, t.predict(Pte)) * 100)
    plt.figure(figsize=(6, 4))
    plt.plot(depths, tr_a, "o-", label="Train")
    plt.plot(depths, te_a, "o-", label="Test")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy (%)")
    plt.title("Decision Tree: depth vs overfitting")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/4_tree_depth_sweep.png", dpi=150)
    plt.close()

    # --- ANN learning curve -------------------------------------------------
    ann = models["ANN"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ann.loss_curve_)
    ax[0].set(xlabel="Epoch", ylabel="Training loss", title="ANN: loss per epoch")
    ax[1].plot(np.array(ann.validation_scores_) * 100)
    ax[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="ANN: validation accuracy")
    for a_ in ax:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/5_ann_learning_curve.png", dpi=150)
    plt.close()

    # --- confusion matrix of best model ---------------------------------
    best = table.sort_values("Test accuracy %").iloc[-1]["Model"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(yte, preds[best], display_labels=classes,
                                            xticks_rotation=45, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {best}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/6_confusion_matrix_{best.replace(' ', '_')}.png", dpi=150)
    plt.close()

    # --- K-Means (unsupervised: NO labels used for clustering; real doodles only) ---
    real = ytr < len(CATEGORIES)
    Xk, yk, Pk = Xtr[real], ytr[real], Ptr[real]
    inertia = [KMeans(k, n_init=3, random_state=42).fit(Pk).inertia_ for k in range(2, 17, 2)]
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 17, 2), inertia, "o-")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia")
    plt.title("K-Means: elbow method")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/7_kmeans_elbow.png", dpi=150)
    plt.close()

    km = KMeans(n_clusters=len(CATEGORIES), n_init=5, random_state=42).fit(Pk)
    ari = adjusted_rand_score(yk, km.labels_)
    hom = homogeneity_score(yk, km.labels_)
    print(f"K-Means (k={len(CATEGORIES)}): Adjusted Rand Index {ari:.3f}, homogeneity {hom:.3f}")
    fig, ax = plt.subplots(2, 5, figsize=(11, 5))
    for c, a_ in enumerate(ax.ravel()):
        members = np.where(km.labels_ == c)[0]
        counts = np.bincount(yk[members], minlength=len(CATEGORIES))
        a_.imshow(Xk[members].mean(axis=0).reshape(28, 28), cmap="gray_r")
        a_.set_title(f"Cluster {c}: {CATEGORIES[counts.argmax()]}\n{counts.max() / len(members):.0%} pure",
                     fontsize=9)
        a_.axis("off")
    plt.suptitle("K-Means: average doodle in each cluster (learned without labels)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/8_kmeans_clusters.png", dpi=150)
    plt.close()

    # --- PCA 2D scatter ----------------------------------------------------
    p2 = PCA(n_components=2, random_state=42).fit_transform(Xtr[:4000])
    plt.figure(figsize=(7, 6))
    for c in range(len(classes)):
        m = ytr[:4000] == c
        plt.scatter(p2[m, 0], p2[m, 1], s=5, label=classes[c])
    plt.legend(markerscale=3, fontsize=8)
    plt.title("PCA: doodles projected to 2D")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/9_pca_scatter.png", dpi=150)
    plt.close()

    # --- "attempt failed" rules: how well do they work? ---------------------
    P = np.mean([m.predict_proba(Pte) for m in models.values()], axis=0)
    conf, top = P.max(axis=1), P.argmax(axis=1)
    is_real = yte < len(CATEGORIES)
    not_scrib = top < len(CATEGORIES)            # model did not say "scribble"
    ths = np.arange(0.30, 0.96, 0.05)

    def accepted_mask(t):
        return (conf >= t) & not_scrib

    acc_ok = [(top[accepted_mask(t) & is_real] == yte[accepted_mask(t) & is_real]).mean() * 100
              if (accepted_mask(t) & is_real).any() else np.nan for t in ths]
    real_acc = [accepted_mask(t)[is_real].mean() * 100 for t in ths]
    plt.figure(figsize=(6.5, 4))
    plt.plot(ths, acc_ok, "o-", label="Accuracy on accepted real doodles (%)")
    plt.plot(ths, real_acc, "o-", label="Real doodles accepted (%)")
    if (~is_real).any():
        plt.plot(ths, [(~accepted_mask(t))[~is_real].mean() * 100 for t in ths], "o-",
                 label="Scribbles rejected (%)")
    plt.axvline(CONFIDENCE_MIN, color="gray", ls="--", label=f"our threshold = {CONFIDENCE_MIN}")
    plt.xlabel("Confidence threshold")
    plt.title("Rejecting unclear drawings ('attempt failed')")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/10_reject_threshold.png", dpi=150)
    plt.close()
    keep = accepted_mask(CONFIDENCE_MIN)
    print(f"Threshold {CONFIDENCE_MIN}: accepts {keep[is_real].mean() * 100:.1f}% of real doodles "
          f"(accuracy {(top[keep & is_real] == yte[keep & is_real]).mean() * 100:.1f}%)"
          + (f", rejects {(~keep)[~is_real].mean() * 100:.1f}% of scribbles" if (~is_real).any() else ""))

    joblib.dump({"pca": pca, "models": models, "classes": classes}, MODEL_FILE)
    print(f"\nSaved models to {MODEL_FILE} and graphs/tables to the '{OUT_DIR}' folder.")
    print("Now run:  python doodle_guesser.py app")


# ----------------------------------------------------------------------------
# 4. PREDICTION + DRAWING APP
# ----------------------------------------------------------------------------
def predict(bundle, pil_img):
    """Look at a drawing (white strokes on black). Returns a dict with a 'verdict':
    'empty' / 'tiny' / 'unsure' (attempt failed) / 'ok', the guess and all probabilities."""
    raw = np.array(pil_img.convert("L"))
    ys, xs = np.where(raw > 40)
    if len(xs) == 0:
        return {"verdict": "empty"}
    if max(ys.max() - ys.min(), xs.max() - xs.min()) < MIN_DRAW_SIZE:
        return {"verdict": "tiny"}
    f = bundle["pca"].transform(normalize(raw).reshape(1, -1).astype(np.float32) / 255.0)
    probs = {name: m.predict_proba(f)[0] for name, m in bundle["models"].items()}
    final = np.mean(list(probs.values()), axis=0)                  # soft voting
    top = int(np.argmax(final))
    votes = sum(int(np.argmax(p)) == top for p in probs.values())  # how many models agree
    probs["Final guess"] = final
    ok = final[top] >= CONFIDENCE_MIN and votes >= MIN_AGREE and bundle["classes"][top] != SCRIBBLE
    return {"verdict": "ok" if ok else "unsure", "guess": bundle["classes"][top],
            "probs": probs, "votes": votes}


def a_or_an(word):
    return ("an " if word[0] in "aeiou" else "a ") + word


SIZE, BRUSH = 300, 14
GREEN, ORANGE, RED = "#1a7f37", "#c45c00", "#b42318"


class DoodleApp:
    def __init__(self, root, bundle):
        import tkinter as tk
        from PIL import ImageDraw
        self.tk, self.root, self.b = tk, root, bundle
        self.classes = bundle["classes"]
        self.targets = [c for c in self.classes if c != SCRIBBLE]
        self.score, self.round, self.solved = 0, 0, False
        self.last, self.target, self.after_id = None, None, None
        self.img = Image.new("L", (SIZE, SIZE), 0)
        self.pen = ImageDraw.Draw(self.img)
        root.title("Doodle Guesser")

        left = tk.Frame(root)
        left.pack(side="left", padx=12, pady=12)
        right = tk.Frame(root, width=340)
        right.pack(side="left", padx=12, pady=12, anchor="n")

        self.canvas = tk.Canvas(left, width=SIZE, height=SIZE, bg="black", highlightthickness=2,
                                highlightbackground="#888", cursor="pencil")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        btns = tk.Frame(left)
        btns.pack(pady=8)
        tk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=4)
        tk.Button(btns, text="Next challenge", command=self.next_challenge).pack(side="left", padx=4)
        self.challenge = tk.BooleanVar(value=True)
        tk.Checkbutton(btns, text="Challenge mode", variable=self.challenge,
                       command=self.next_challenge).pack(side="left", padx=4)

        self.title = tk.Label(right, text="", font=("Helvetica", 22, "bold"), wraplength=330,
                              justify="left", anchor="w")
        self.title.pack(anchor="w")
        self.score_lbl = tk.Label(right, text="", font=("Helvetica", 13))
        self.score_lbl.pack(anchor="w")
        self.message = tk.Label(right, text="", font=("Helvetica", 18, "bold"), wraplength=330,
                                justify="left", anchor="nw", height=4, width=26)
        self.message.pack(anchor="w", pady=8)
        tk.Label(right, text="I know these: " + ", ".join(self.targets), font=("Helvetica", 11),
                 fg="#555", wraplength=330, justify="left").pack(anchor="w", pady=(0, 8))

        # hidden by default: the technical details (for teachers / viva)
        self.details_on = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Show AI details (for teachers)", variable=self.details_on,
                       command=self.toggle_details).pack(anchor="w")
        self.details = tk.Frame(right)
        self.lbls = {}
        for name in ["KNN", "Decision Tree", "ANN", "Final guess"]:
            lb = tk.Label(self.details, text="", font=("Menlo", 12), anchor="w", justify="left")
            lb.pack(anchor="w")
            self.lbls[name] = lb
        self.bars = tk.Canvas(self.details, width=320, height=140, highlightthickness=0)
        self.bars.pack(pady=6)
        self.next_challenge()

    # --- helpers -----------------------------------------------------------
    def say(self, text, colour):
        self.message.config(text=text, fg=colour)

    def toggle_details(self):
        if self.details_on.get():
            self.details.pack(anchor="w", pady=6)
        else:
            self.details.pack_forget()

    def show_details(self, res):
        self.bars.delete("all")
        for lb in self.lbls.values():
            lb.config(text="")
        if not res or "probs" not in res:
            return
        for name, probs in res["probs"].items():
            i = int(np.argmax(probs))
            self.lbls[name].config(text=f"{name + ':':14s} {self.classes[i]} ({probs[i] * 100:.0f}%)")
        final = res["probs"]["Final guess"]
        for row, i in enumerate(np.argsort(final)[::-1][:5]):
            y = 8 + row * 26
            self.bars.create_text(2, y + 8, text=self.classes[i], anchor="w", font=("Helvetica", 11))
            self.bars.create_rectangle(90, y, 90 + final[i] * 220, y + 16, fill="#4c8bf5", outline="")
            self.bars.create_text(94 + final[i] * 220, y + 8, text=f"{final[i] * 100:.0f}%",
                                  anchor="w", font=("Helvetica", 10))
        self.bars.create_text(2, 136, anchor="sw", font=("Helvetica", 10),
                              text=f"{res['votes']}/3 models agree | needs {CONFIDENCE_MIN:.0%} "
                                   f"and {MIN_AGREE}/3 agreement")

    # --- drawing -----------------------------------------------------------
    def on_down(self, e):
        self.last = (e.x, e.y)
        r = BRUSH / 2
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white", outline="white")
        self.pen.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def on_move(self, e):
        if self.last is None:
            return
        x0, y0 = self.last
        self.canvas.create_line(x0, y0, e.x, e.y, fill="white", width=BRUSH, capstyle="round")
        self.pen.line([x0, y0, e.x, e.y], fill=255, width=BRUSH)
        r = BRUSH / 2
        self.pen.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
        self.last = (e.x, e.y)

    def on_up(self, _):
        self.last = None
        self.guess()                                   # guess every time you lift the pen

    def clear(self):
        self.canvas.delete("all")
        self.pen.rectangle([0, 0, SIZE, SIZE], fill=0)
        self.show_details(None)
        self.say("Draw it big, in the middle!", "#333")

    # --- game logic --------------------------------------------------------
    def next_challenge(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.clear()
        self.solved = False
        if self.challenge.get():
            self.round += 1
            self.target = random.choice(self.targets)
            self.title.config(text=f"Can you draw {a_or_an(self.target)}?")
            self.score_lbl.config(text=f"Score: {self.score}   |   Challenge {self.round}")
        else:
            self.target = None
            self.title.config(text="Draw anything you like!")
            self.score_lbl.config(text="")

    def guess(self):
        res = predict(self.b, self.img)
        v = res["verdict"]
        if v == "empty" or self.solved:
            return
        self.show_details(res)
        if v == "tiny":
            self.say("Please draw it bigger!\nI can't see it yet.", ORANGE)
        elif v == "unsure":
            self.say("Oops! Attempt failed.\nI can't tell what this is.\nTry again!", RED)
        else:
            guess = res["guess"]
            if self.target is None:
                self.say(f"I think it is {a_or_an(guess)}!", GREEN)
            elif guess == self.target:
                self.solved = True
                self.score += 1
                self.say(f"Great job!\nI can see {a_or_an(guess)}!", GREEN)
                self.score_lbl.config(text=f"Score: {self.score}   |   Challenge {self.round}")
                self.after_id = self.root.after(2200, self.next_challenge)
            else:
                self.say(f"Hmm, I see {a_or_an(guess)}.\nCan you draw {a_or_an(self.target)}?", ORANGE)


def run_app():
    import joblib
    import tkinter as tk
    if not os.path.exists(MODEL_FILE):
        sys.exit("Model not found. Run:  python doodle_guesser.py train")
    bundle = joblib.load(MODEL_FILE)
    root = tk.Tk()
    DoodleApp(root, bundle)
    root.mainloop()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["download", "train", "app"])
    p.add_argument("--samples", type=int, default=3000, help="doodles per category")
    p.add_argument("--pca", type=int, default=50, help="PCA components")
    p.add_argument("--scribbles", type=int, default=3000,
                   help="random scribbles for the 'not a doodle' class (0 = off)")
    a = p.parse_args()
    if a.mode == "download":
        download(a.samples)
    elif a.mode == "train":
        train(a)
    else:
        run_app()
