"""
Stage 2.5: LDA Topic Modeling

Trains an LDA (Latent Dirichlet Allocation) topic model to generate topic features:
- lda_t1 through lda_t5: Topic probability distributions for each review

Topic modeling captures the underlying themes in reviews (e.g., food quality,
service, atmosphere). These topic probabilities become features for the final model.

Input: Stage 2.4 output (FastText features + review_text)
Pipeline: 2.1 → 2.2 → 2.3 → 2.4 → 2.5 (this stage - FINAL NLP)
Output: Parquet files with lda_t1-lda_t5 columns added

Note: Uses Gensim's LdaMulticore for efficient training.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
import spacy
from nltk.corpus import stopwords
import nltk

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Number of topics
NUM_TOPICS = 5


def load_spacy_model():
    """Load spaCy model for lemmatization."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        logger.info("Loaded spaCy model")
        return nlp
    except OSError:
        logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


def preprocess_texts(texts: list, nlp, stop_words: set) -> list:
    """Preprocess texts for LDA.

    Args:
        texts: List of review texts
        nlp: spaCy model
        stop_words: Set of stopwords to remove

    Returns:
        List of preprocessed token lists
    """
    logger.info(f"Preprocessing {len(texts):,} texts...")

    # Tokenize and remove stopwords
    processed = []
    for text in texts:
        if not text:
            processed.append([])
            continue

        # Simple preprocessing: tokenize, lowercase, remove short words
        tokens = simple_preprocess(str(text), deacc=True)

        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatize with spaCy (keep nouns, verbs, adjectives, adverbs)
        doc = nlp(" ".join(tokens))
        allowed_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        tokens = [token.lemma_ for token in doc if token.pos_ in allowed_pos]

        processed.append(tokens)

    logger.info("Preprocessing complete")
    return processed


def train_lda_model(corpus: list, id2word: corpora.Dictionary, model_path: Path) -> LdaMulticore:
    """Train LDA model.

    Args:
        corpus: Bag-of-words corpus
        id2word: Dictionary mapping
        model_path: Path to save model

    Returns:
        Trained LDA model
    """
    logger.info(f"Training LDA model with {NUM_TOPICS} topics...")

    model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=NUM_TOPICS,
        passes=5,
        workers=4,
        random_state=42,
    )

    # Save model
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Log topics
    logger.info("Top words per topic:")
    for idx, topic in model.print_topics(-1):
        logger.info(f"  Topic {idx + 1}: {topic}")

    return model


def get_topic_features(model: LdaMulticore, corpus: list) -> pd.DataFrame:
    """Get topic probability features for each document.

    Args:
        model: Trained LDA model
        corpus: Bag-of-words corpus

    Returns:
        DataFrame with lda_t1 through lda_t5 columns
    """
    topic_dicts = []
    for doc_bow in corpus:
        # Get topic distribution (ensure all topics are represented)
        topics = model.get_document_topics(doc_bow, minimum_probability=0.0)
        topic_dict = {f"lda_t{i+1}": round(prob, 4) for i, (_, prob) in enumerate(sorted(topics))}
        topic_dicts.append(topic_dict)

    return pd.DataFrame(topic_dicts)


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 2.5: LDA Topic Modeling")
    logger.info("=" * 60)

    # Setup paths
    input_dir = PathConfig.get_nlp_embeddings_dir()  # From 2.4 (FastText features)
    output_dir = PathConfig.get_nlp_lda_dir()
    model_dir = PathConfig.get_nlp_models_dir()

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model directory: {model_dir}")

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load spaCy and stopwords
    nlp = load_spacy_model()
    stop_words = set(stopwords.words('english'))

    # Load training data
    train_df = pd.read_parquet(input_dir / "train.parquet")
    logger.info(f"Loaded train: {len(train_df):,} rows")

    # Preprocess training texts
    train_texts = train_df["review_text"].fillna("").tolist()
    processed_train = preprocess_texts(train_texts, nlp, stop_words)

    # Create dictionary and corpus
    logger.info("Creating dictionary and corpus...")
    id2word = corpora.Dictionary(processed_train)
    id2word.filter_extremes(no_below=5, no_above=0.5)  # Filter rare/common words
    train_corpus = [id2word.doc2bow(text) for text in processed_train]
    logger.info(f"Dictionary size: {len(id2word):,} tokens")

    # Train model
    model_path = model_dir / "lda_model"
    model = train_lda_model(train_corpus, id2word, model_path)

    # Save dictionary for later use
    id2word.save(str(model_dir / "lda_dictionary"))
    logger.info("Dictionary saved")

    # Process each split
    total_records = 0
    for split in ["train", "test", "holdout"]:
        logger.info(f"Processing {split} split...")

        # Load data
        df = pd.read_parquet(input_dir / f"{split}.parquet")

        # Preprocess texts
        texts = df["review_text"].fillna("").tolist()
        processed = preprocess_texts(texts, nlp, stop_words)

        # Create corpus using trained dictionary
        corpus = [id2word.doc2bow(text) for text in processed]

        # Get topic features
        topic_df = get_topic_features(model, corpus)

        # Add topic columns to original DataFrame
        df = pd.concat([df.reset_index(drop=True), topic_df], axis=1)

        # Save output
        output_path = output_dir / f"{split}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"  Saved {len(df):,} rows to {output_path}")

        total_records += len(df)

    logger.info("=" * 60)
    logger.info(f"Completed! Processed {total_records:,} total records")
    logger.info(f"Features added: lda_t1, lda_t2, lda_t3, lda_t4, lda_t5")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
