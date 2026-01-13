"""
Stage 2.2: NLP Linguistic Features with spaCy

Extracts linguistic features using spaCy:
- Token count, stopword count/percentage
- Entity count/percentage
- Part-of-speech (POS) tags - 17 types
- Dependency relations - 46 types
- Named entity types - 18 types

Input: Stage 2.1 output (with basic text features)
Output: Parquet files with linguistic features added

Note: Uses pandas + spaCy (not Spark) for nlp.pipe() efficiency.
Processes in chunks to manage memory.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import spacy

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Linguistic feature lists (Universal Dependencies)
POS_LIST = [
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]

DEP_LIST = [
    "ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos",
    "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj",
    "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj",
    "mark", "meta", "neg", "nmod", "npadvmod", "nsubj", "nsubjpass",
    "nummod", "oprd", "parataxis", "pcomp", "pobj", "poss", "preconj",
    "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp",
]

ENT_LIST = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC",
    "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT",
    "QUANTITY", "TIME", "WORK_OF_ART",
]


def load_spacy_model():
    """Load spaCy English model."""
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
        return nlp
    except OSError:
        logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


def extract_spacy_features(df: pd.DataFrame, nlp, text_col: str = "review_text") -> pd.DataFrame:
    """Extract spaCy linguistic features from text.

    Args:
        df: DataFrame with text column
        nlp: spaCy language model
        text_col: Name of text column

    Returns:
        DataFrame with linguistic features added (text column removed)
    """
    # Process texts with spaCy pipe (efficient batch processing)
    texts = df[text_col].fillna("").tolist()
    docs = list(nlp.pipe(texts, n_process=1, batch_size=100))

    # Basic counts
    df["token_count"] = [len(doc) for doc in docs]
    df["stopword_count"] = [len([t for t in doc if t.is_stop]) for doc in docs]
    df["stopword_perc"] = [
        round(len([t for t in doc if t.is_stop]) / max(len(doc), 1), 5)
        for doc in docs
    ]
    df["ent_count"] = [len(doc.ents) for doc in docs]
    df["ent_perc"] = [
        round(len(doc.ents) / max(len(doc), 1), 5)
        for doc in docs
    ]

    # POS tag features
    for pos in POS_LIST:
        col_count = f"pos_{pos.lower()}_count"
        col_perc = f"pos_{pos.lower()}_perc"
        df[col_count] = [len([t for t in doc if t.pos_ == pos]) for doc in docs]
        df[col_perc] = [
            round(len([t for t in doc if t.pos_ == pos]) / max(len(doc), 1), 5)
            for doc in docs
        ]

    # Dependency relation features
    for dep in DEP_LIST:
        col_count = f"dep_{dep.lower()}_count"
        col_perc = f"dep_{dep.lower()}_perc"
        df[col_count] = [len([t for t in doc if t.dep_ == dep]) for doc in docs]
        df[col_perc] = [
            round(len([t for t in doc if t.dep_ == dep]) / max(len(doc), 1), 5)
            for doc in docs
        ]

    # Named entity type features
    for ent in ENT_LIST:
        col_count = f"ent_{ent.lower()}_count"
        col_perc = f"ent_{ent.lower()}_perc"
        df[col_count] = [len([e for e in doc.ents if e.label_ == ent]) for doc in docs]
        df[col_perc] = [
            round(len([e for e in doc.ents if e.label_ == ent]) / max(len(doc), 1), 5)
            for doc in docs
        ]

    # Keep text column for downstream stages (2.3, 2.4, 2.5 need it)
    # It will be dropped in the final combine stage

    return df


def process_split(nlp, split_name: str, input_dir: Path, output_dir: Path, chunksize: int = 10000):
    """Process a single data split with chunking.

    Args:
        nlp: spaCy model
        split_name: Name of split ('train', 'test', 'holdout')
        input_dir: Path to Stage 2.1 output
        output_dir: Path to Stage 2.2 output
        chunksize: Number of rows per chunk
    """
    input_path = input_dir / f"{split_name}.parquet"
    output_path = output_dir / f"{split_name}.parquet"

    logger.info(f"Processing {split_name} split...")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")

    # Read full parquet (small test data fits in memory)
    df = pd.read_parquet(input_path)
    record_count = len(df)
    logger.info(f"  Loaded {record_count:,} records")

    # Process in chunks for memory efficiency
    chunks = []
    for i in range(0, len(df), chunksize):
        chunk = df.iloc[i:i + chunksize].copy()
        chunk = extract_spacy_features(chunk, nlp)
        chunks.append(chunk)
        logger.info(f"  Processed rows {i:,} - {min(i + chunksize, len(df)):,}")

    # Combine and write
    result = pd.concat(chunks, ignore_index=True)
    result.to_parquet(output_path, index=False)
    logger.info(f"  Wrote {len(result):,} rows with {len(result.columns)} columns")

    return record_count


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Stage 2.2: NLP Linguistic Features (spaCy)")
    logger.info("=" * 60)

    # Setup paths
    input_dir = PathConfig.get_nlp_basic_dir()
    output_dir = PathConfig.get_nlp_spacy_dir()

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load spaCy model
    nlp = load_spacy_model()

    # Process each split
    total_records = 0
    for split in ["train", "test", "holdout"]:
        count = process_split(nlp, split, input_dir, output_dir)
        total_records += count

    logger.info("=" * 60)
    logger.info(f"Completed! Processed {total_records:,} total records")
    logger.info(f"Features added: {5 + len(POS_LIST)*2 + len(DEP_LIST)*2 + len(ENT_LIST)*2}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
