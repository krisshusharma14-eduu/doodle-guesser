import json, os, random, ssl, string, time
import numpy as np, joblib
from urllib.parse import quote
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from common import OBJECTS, OTHER, preprocess

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    CTX = None

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def fetch(cat, n):
    cache = f"data/{cat}_{n}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    req = Request(BASE + quote(cat) + ".npy",
                  headers={"Range": f"bytes=0-{256 + 784 * n - 1}", "User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            data = urlopen(req, timeout=120, context=CTX).read()
            off = 10 + int.from_bytes(data[8:10], "little")
            arr = np.frombuffer(data[off:], dtype=np.uint8)
            arr = arr[: len(arr) // 784 * 784].reshape(-1, 784)[:n]
            np.save(cache, arr)
            return arr
        except Exception as e:
            print(f"  {cat}: attempt {attempt + 1} failed ({e})")
            time.sleep(2)
    return None


def fit(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True,
                        max_iter=40, random_state=0, verbose=False).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    joblib.dump(clf, f"models/{name}.joblib")
    print(f"{name}: test accuracy {acc:.3f}")
    return acc


# ---------- objects ----------
X, y = [], []
for cat in OBJECTS:
    print("downloading", cat)
    arr = fetch(cat, 3000)
    if arr is None:
        print("  skipped", cat); continue
    X.append(arr / 255.0); y += [cat] * len(arr)
others = [fetch(c, 500) for c in OTHER]
others = [o for o in others if o is not None]
if others:
    o = np.vstack(others); X.append(o / 255.0); y += ["other"] * len(o)
metrics = {"objects": fit(np.vstack(X).astype(np.float32), np.array(y), "objects")}

# ---------- letters (rendered from system fonts) ----------
CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/comic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
fonts = [f for f in CANDIDATES if os.path.exists(f)]


def render(ch):
    size = 160
    img = Image.new("L", (size, size), 0)
    font = ImageFont.truetype(random.choice(fonts), random.randint(90, 130))
    ImageDraw.Draw(img).text((size / 2, size / 2), ch, fill=255, font=font, anchor="mm")
    img = img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR)
    sh = random.uniform(-0.25, 0.25)
    img = img.transform(img.size, Image.AFFINE, (1, sh, -sh * size / 2, 0, 1, 0),
                        resample=Image.BILINEAR)
    k = random.choice([1, 3, 3, 5])
    if k > 1:
        img = img.filter(ImageFilter.MaxFilter(k))
    return preprocess(img)


if fonts:
    X, y = [], []
    for ch in string.ascii_uppercase:
        for _ in range(600):
            v = render(ch)
            if v is not None:
                X.append(v); y.append(ch)
    metrics["letters"] = fit(np.array(X, dtype=np.float32), np.array(y), "letters")
else:
    print("No fonts found - letters mode skipped.")

json.dump(metrics, open("models/metrics.json", "w"))