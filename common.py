import numpy as np
from PIL import Image

USE_EMOJI = True    # set to False if the picture cards look broken on your computer

OBJECTS = ["apple", "cat", "house", "car", "tree", "sun", "fish", "star",
           "umbrella", "clock", "heart", "flower", "bicycle", "airplane", "cup",
           "dog", "banana", "moon", "cloud", "rainbow", "butterfly", "rabbit",
           "pizza", "ice cream", "strawberry", "bird"]
OTHER = ["book", "chair", "hat", "key", "lightning",
         "mountain", "pencil", "door", "table", "scissors"]   # trained as "other" -> Attempt failed

# picture card for every object (to add a new object: add it to OBJECTS and here)
EMOJI = {"apple": "🍎", "cat": "🐱", "house": "🏠", "car": "🚗", "tree": "🌳", "sun": "☀️",
         "fish": "🐟", "star": "⭐", "umbrella": "☂️", "clock": "🕒", "heart": "❤️",
         "flower": "🌸", "bicycle": "🚲", "airplane": "✈️", "cup": "☕", "dog": "🐶",
         "banana": "🍌", "moon": "🌙", "cloud": "☁️", "rainbow": "🌈", "butterfly": "🦋",
         "rabbit": "🐰", "pizza": "🍕", "ice cream": "🍦", "strawberry": "🍓", "bird": "🐦"}

LETTER_WORDS = {"A": "Apple", "B": "Ball", "C": "Cat", "D": "Dog", "E": "Elephant",
                "F": "Fish", "G": "Grapes", "H": "House", "I": "Ice cream", "J": "Jug",
                "K": "Kite", "L": "Lion", "M": "Mango", "N": "Nest", "O": "Orange",
                "P": "Parrot", "Q": "Queen", "R": "Rabbit", "S": "Sun", "T": "Tree",
                "U": "Umbrella", "V": "Van", "W": "Watch", "X": "Xylophone",
                "Y": "Yak", "Z": "Zebra"}
LETTER_EMOJI = {"A": "🍎", "B": "⚽", "C": "🐱", "D": "🐶", "E": "🐘", "F": "🐟", "G": "🍇",
                "H": "🏠", "I": "🍦", "J": "🏺", "K": "🪁", "L": "🦁", "M": "🥭", "N": "🪺",
                "O": "🍊", "P": "🦜", "Q": "👸", "R": "🐰", "S": "☀️", "T": "🌳", "U": "☂️",
                "V": "🚐", "W": "⌚", "X": "🎶", "Y": "🐂", "Z": "🦓"}

MIN_CONF = 0.55     # raise = stricter (more "Attempt failed")
MIN_MARGIN = 0.15   # gap needed between 1st and 2nd guess


def pic(e):
    """Return the emoji, or nothing if USE_EMOJI is False."""
    return e if USE_EMOJI else ""


def preprocess(img):
    """PIL 'L' image (white on black) -> 784 floats, or None if nothing/too small."""
    a = np.array(img)
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return None
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = img.size
    if max(w, h) < 25:
        return None
    s = max(w, h)
    sq = Image.new("L", (s, s), 0)
    sq.paste(img, ((s - w) // 2, (s - h) // 2))
    small = sq.resize((28, 28), Image.LANCZOS)
    return np.asarray(small, dtype=np.float32).ravel() / 255.0


def decide(classes, probs):
    order = np.argsort(probs)[::-1]
    top = [(str(classes[i]), float(probs[i])) for i in order[:3]]
    label, conf = top[0]
    margin = conf - top[1][1]
    if label == "other" or conf < MIN_CONF or margin < MIN_MARGIN:
        return None, conf, top
    return label, conf, top