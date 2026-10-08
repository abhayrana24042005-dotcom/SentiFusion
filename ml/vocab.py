import json
import re
from typing import List, Dict, Optional, Set

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# Broad initial vocabulary covering sentiment polarity, intensity, common descriptors, nouns, and modalities
DEFAULT_VOCAB = [
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
    # Positive sentiment words
    "good", "great", "excellent", "amazing", "love", "loved", "loving", "awesome", "fantastic", "wonderful", "perfect",
    "happy", "joy", "joyful", "beautiful", "superb", "brilliant", "delighted", "delightful", "best", "positive",
    "impressive", "outstanding", "satisfaction", "satisfied", "satisfying", "pleasant", "pleased", "splendid",
    "charming", "super", "fabulous", "terrific", "terrific", "phenomenal", "exceptional", "stellar", "flawless",
    "gorgeous", "clean", "smooth", "fast", "reliable", "premium", "top", "worth", "recommend", "recommended",
    "favorite", "fun", "cool", "magic", "magical", "vibrant", "bright", "exciting", "excited", "friendly", "helpful",
    "warm", "sweet", "nice", "fine", "enjoyed", "enjoyable", "glad", "proud", "thankful", "grateful", "success",
    
    # Negative sentiment words
    "bad", "terrible", "horrible", "awful", "hate", "hated", "hating", "worst", "poor", "poorest", "disappointed",
    "disappointing", "disappointment", "ugly", "broken", "useless", "annoying", "annoyed", "frustrated", "frustrating",
    "sad", "angry", "negative", "slow", "pathetic", "waste", "defective", "faulty", "garbage", "trash", "junk",
    "horrid", "nasty", "dreadful", "miserable", "painful", "worthless", "failed", "failure", "failing", "damaged",
    "cheap", "flimsy", "regret", "regretted", "toxic", "rude", "delayed", "late", "ruined", "crash", "crashed",
    "scam", "shame", "unpleasant", "unusable", "terribly", "awfully", "poorly", "worse", "disaster", "mess",
    
    # Neutral / Contextual words
    "neutral", "okay", "ok", "average", "moderate", "standard", "ordinary", "expected", "regular", "normal",
    "typical", "fair", "passable", "acceptable", "routine", "common", "plain", "simple", "adequate", "medium",
    
    # Common functional / grammatical words
    "the", "is", "are", "was", "were", "a", "an", "this", "that", "these", "those", "it", "its", "and", "or",
    "but", "with", "without", "for", "to", "at", "by", "from", "in", "on", "into", "of", "about", "as", "if",
    "very", "so", "really", "quite", "somewhat", "too", "extremely", "highly", "completely", "totally", "absolutely",
    "definitely", "fairly", "hardly", "barely", "almost", "just", "only", "always", "never", "sometimes", "often",
    "not", "no", "never", "cannot", "cant", "dont", "wont", "didnt", "isnt", "arent", "wasnt", "werent",
    
    # Entity, product, and review domain words
    "product", "quality", "service", "experience", "food", "item", "movie", "book", "delivery", "design", "order",
    "customer", "support", "package", "box", "shipping", "price", "value", "feature", "features", "camera", "photo",
    "image", "picture", "screen", "display", "battery", "performance", "build", "material", "sound", "app", "ui",
    "software", "device", "unit", "hardware", "arrived", "received", "purchased", "bought", "works", "working",
    "color", "size", "feel", "look", "looks", "looking", "seems", "used", "using", "tried", "tested", "overall"
]


class TextTokenizer:
    """
    Robust tokenizer and vocabulary management for sentiment representations.
    Supports dynamic vocabulary expansion when new training cases are introduced.
    """
    def __init__(self, vocab_list: Optional[List[str]] = None):
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        initial_words = DEFAULT_VOCAB if vocab_list is None else vocab_list
        for word in initial_words:
            self.add_word(word)

    def clean_text(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        # Normalize common elongated characters (e.g. "wonderfull" -> "wonderful", "amazzzing" -> "amazing", "soooo" -> "so")
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        
        # Common review typos mapping
        typo_fixes = {
            "wonderfull": "wonderful",
            "awfull": "awful",
            "terribel": "terrible",
            "beautifull": "beautiful",
            "loove": "love",
            "happpy": "happy",
            "disapointed": "disappointed",
            "dissapointed": "disappointed",
            "dissappointed": "disappointed",
            "usefull": "useful",
            "perfict": "perfect",
            "greatt": "great"
        }
        for wrong, right in typo_fixes.items():
            text = re.sub(r"\b" + wrong + r"\b", right, text)

        # Keep letters, digits, and common punctuation
        text = re.sub(r"[^a-z0-9\s!?,.'-]", " ", text)
        # Separate punctuation into distinct tokens
        text = re.sub(r"([!?,.])", r" \1 ", text)
        tokens = [t.strip() for t in text.split() if t.strip()]
        return tokens

    def tokenize(self, text: str) -> List[str]:
        return self.clean_text(text)

    def add_word(self, word: str) -> int:
        word = word.lower()
        if word not in self.word2idx:
            new_idx = len(self.word2idx)
            self.word2idx[word] = new_idx
            self.idx2word[new_idx] = word
            return new_idx
        return self.word2idx[word]

    def build_vocab_from_texts(self, texts: List[str]) -> int:
        added_count = 0
        for text in texts:
            tokens = self.clean_text(text)
            for token in tokens:
                if token not in self.word2idx:
                    self.add_word(token)
                    added_count += 1
        return added_count

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = self.clean_text(text)
        token_ids = [SOS_IDX]
        for token in tokens:
            token_ids.append(self.word2idx.get(token, UNK_IDX))
        token_ids.append(EOS_IDX)

        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len - 1] + [EOS_IDX]
        else:
            token_ids = token_ids + [PAD_IDX] * (max_len - len(token_ids))

        return token_ids

    def get_lexical_sentiment(self, text: str) -> str:
        tokens = self.clean_text(text)
        pos_words = {
            "good", "great", "excellent", "amazing", "love", "loved", "loving", "awesome", "fantastic", "wonderful", "perfect",
            "happy", "joy", "joyful", "beautiful", "superb", "brilliant", "delighted", "delightful", "best", "positive",
            "impressive", "outstanding", "satisfaction", "satisfied", "satisfying", "pleasant", "pleased", "splendid",
            "charming", "super", "fabulous", "terrific", "phenomenal", "exceptional", "stellar", "flawless",
            "gorgeous", "clean", "smooth", "fast", "reliable", "premium", "top", "worth", "recommend", "recommended",
            "favorite", "fun", "cool", "magic", "magical", "vibrant", "bright", "exciting", "excited", "friendly", "helpful",
            "warm", "sweet", "nice", "fine", "enjoyed", "enjoyable", "glad", "proud", "thankful", "grateful", "success", "flawless"
        }
        neg_words = {
            "bad", "terrible", "horrible", "awful", "hate", "hated", "hating", "worst", "poor", "poorest", "disappointed",
            "disappointing", "disappointment", "ugly", "broken", "useless", "annoying", "annoyed", "frustrated", "frustrating",
            "sad", "angry", "negative", "slow", "pathetic", "waste", "defective", "faulty", "garbage", "trash", "junk",
            "horrid", "nasty", "dreadful", "miserable", "painful", "worthless", "failed", "failure", "failing", "damaged",
            "cheap", "flimsy", "regret", "regretted", "toxic", "rude", "delayed", "late", "ruined", "crash", "crashed",
            "scam", "shame", "unpleasant", "unusable", "terribly", "awfully", "poorly", "worse", "disaster", "mess"
        }
        pos_cnt = sum(1 for t in tokens if t in pos_words)
        neg_cnt = sum(1 for t in tokens if t in neg_words)
        if pos_cnt > neg_cnt:
            return "Positive"
        elif neg_cnt > pos_cnt:
            return "Negative"
        return "Neutral"

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            if tid in (PAD_IDX, SOS_IDX):
                continue
            if tid == EOS_IDX:
                break
            words.append(self.idx2word.get(tid, UNK_TOKEN))
        return " ".join(words)

    def save(self, filepath: str):
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TextTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(vocab_list=[])
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
