import re
import random
from typing import List
import nltk
from nltk.corpus import wordnet

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def parse_post(text: str) -> tuple:
    """Parse post into title and main text."""
    title_match = re.search(r'\[title\](.*?)\[text\]', text, re.DOTALL)
    main_text_match = re.search(r'\[text\](.*)', text, re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else ""
    main_text = main_text_match.group(1).strip() if main_text_match else ""
    
    return title, main_text


def reconstruct_post(title: str, main_text: str) -> str:
    """Reconstruct post from title and main text."""
    return f"[title]{title} [text]{main_text}"


def synonym_replacement(text: str, n: int = 2) -> str:
    """Replace n random words with synonyms."""
    words = text.split()
    new_words = words.copy()
    
    # Get words that have synonyms
    random_word_list = list(set([
        word for word in words if wordnet.synsets(word)
    ]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if not synonyms:
            continue
            
        synonym_words = set()
        for syn in synonyms:
            for lemma in syn.lemmas():
                synonym_words.add(lemma.name())
        synonym_words.discard(random_word)
        
        if len(synonym_words) > 0:
            synonym = random.choice(list(synonym_words))
            new_words = [
                synonym if word == random_word else word 
                for word in new_words
            ]
            num_replaced += 1
            
        if num_replaced >= n:
            break
            
    return ' '.join(new_words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) <= 1:
        return text
        
    new_words = [w for w in words if random.random() > p]
    if not new_words:
        return random.choice(words)
        
    return " ".join(new_words)


def random_swap(text: str, n: int = 1) -> str:
    """Randomly swap n pairs of words."""
    words = text.split()
    if len(words) < 2:
        return text
        
    for _ in range(n):
        i1, i2 = random.sample(range(len(words)), 2)
        words[i1], words[i2] = words[i2], words[i1]
        
    return " ".join(words)


class TextAugmenter:
    """Text augmentation with configurable strategies."""
    
    def __init__(self, config: dict):
        self.augmentations = config.get('augmentations', [
            'synonym_replacement', 'random_deletion', 'random_swap'
        ])
        self.params = config.get('augmentation_params', {})
        
        self.aug_functions = {
            'synonym_replacement': lambda t: synonym_replacement(
                t, **self.params.get('synonym_replacement', {'n': 2})
            ),
            'random_deletion': lambda t: random_deletion(
                t, **self.params.get('random_deletion', {'p': 0.1})
            ),
            'random_swap': lambda t: random_swap(
                t, **self.params.get('random_swap', {'n': 1})
            ),
        }
    
    def augment_post(self, original_post: str) -> str:
        """Augment a post by applying a random augmentation."""
        title, main_text = parse_post(original_post)
        
        # Choose random augmentation
        aug_name = random.choice(self.augmentations)
        aug_fn = self.aug_functions[aug_name]
        
        # Apply to title and main text
        aug_title = aug_fn(title)
        aug_main_text = aug_fn(main_text)
        
        return reconstruct_post(aug_title, aug_main_text)
