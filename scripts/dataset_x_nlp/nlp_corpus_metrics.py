import math
import spacy
from collections import defaultdict, Counter

# Load the spaCy model globally. Download if not present.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    print("Downloading en_core_web_sm spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def compute_corpus_metrics(corpus: list[str]) -> dict:
    """
    Computes 6 NLP metrics over a corpus of strings.
    
    Args:
        corpus: A list of raw text strings.
        
    Returns:
        A dictionary containing the 6 computed metrics.
    """
    
    # Metric 1: Morphological Complexity (M)
    lemma_to_surface = defaultdict(set)
    
    # Metric 2: Syntactic Entropy (S)
    dep_counts = Counter()
    total_deps = 0
    
    # Metric 3: Redundancy (R)
    vocab = {}
    next_id = 0
    unigram_counts = Counter()
    trigram_counts = Counter()
    gapped_bigram_counts = Counter()
    total_unigrams = 0
    total_trigrams = 0
    
    # Metrics 4, 5, 6
    total_word_len = 0
    total_capitalized = 0
    total_words = 0
    unique_words = set()
    
    # We use nlp.pipe for efficient batch processing of the corpus
    for doc in nlp.pipe(corpus):
        doc_ids = []
        
        for token in doc:
            # 1. Morphological Complexity
            lemma_to_surface[token.lemma_].add(token.text)
            
            # 2. Syntactic Entropy
            dep_counts[token.dep_] += 1
            total_deps += 1
            
            # 3. Redundancy setup
            w_lower = token.text.lower()
            if w_lower not in vocab:
                vocab[w_lower] = next_id
                next_id += 1
            w_id = vocab[w_lower]
            doc_ids.append(w_id)
            unigram_counts[w_id] += 1
            total_unigrams += 1
            
            # 4, 5, 6 setup (filtering out pure punctuation and whitespace)
            if not token.is_punct and not token.is_space:
                total_words += 1
                total_word_len += len(token.text)
                
                # Check capitalization
                if token.text and token.text[0].isupper():
                    total_capitalized += 1
                    
                unique_words.add(token.text.lower())
                
        # 3. Redundancy (extracting trigrams and gapped bigrams per doc)
        for i in range(1, len(doc_ids) - 1):
            w_prev = doc_ids[i-1]
            w_curr = doc_ids[i]
            w_next = doc_ids[i+1]
            
            trigram_counts[(w_prev, w_curr, w_next)] += 1
            gapped_bigram_counts[(w_prev, w_next)] += 1
            total_trigrams += 1

    # --- Compute Final Scalar Values ---

    # Metric 1: Morphological Complexity (M)
    M = sum(len(forms) for forms in lemma_to_surface.values()) / len(lemma_to_surface) if lemma_to_surface else 0.0

    # Metric 2: Syntactic Entropy (S)
    S = 0.0
    if total_deps > 0:
        for count in dep_counts.values():
            p = count / total_deps
            S -= p * math.log(p, 2)  # Base 2 entropy (bits)

    # Metric 3: Redundancy (R)
    R = 0.0
    if total_trigrams > 0:
        sum_pmi = 0.0
        for (w_prev, w_curr, w_next), count in trigram_counts.items():
            p_joint = count / total_trigrams
            p_wi = unigram_counts[w_curr] / total_unigrams
            p_context = gapped_bigram_counts[(w_prev, w_next)] / total_trigrams
            
            # PMI = log( P(w_prev, w_curr, w_next) / (P(w_curr) * P(w_prev, \cdot, w_next)) )
            # Added a tiny epsilon to avoid domain errors if floating point issues occur, 
            # though mathematically p_joint <= p_context and p_wi, so inside is > 0.
            pmi = math.log(p_joint / (p_wi * p_context), 2)
            sum_pmi += pmi * count
            
        R = sum_pmi / total_trigrams

    # Metric 4: Average Word Length (T)
    T = total_word_len / total_words if total_words > 0 else 0.0

    # Metric 5: Capitalization Rate (C)
    C = total_capitalized / total_words if total_words > 0 else 0.0

    # Metric 6: Vocabulary Richness (D)
    D = len(unique_words) / total_words if total_words > 0 else 0.0

    return {
        "Morphological_Complexity": M,
        "Syntactic_Entropy": S,
        "Redundancy": R,
        "Avg_Word_Length": T,
        "Capitalization_Rate": C,
        "Vocabulary_Richness": D
    }

if __name__ == "__main__":
    # Small test suite to verify calculation execution
    sample_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog outpaces a lazy fox.",
        "Dogs are jumping and foxes are running!",
        "Capitalization is Important Here."
    ]
    
    print("Running NLP Corpus Metrics on Sample Corpus...")
    metrics = compute_corpus_metrics(sample_corpus)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
