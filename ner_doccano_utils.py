"""
NER doccano utilites.

Collection of utility functions for during NER preprocessing.
"""

from typing import List, Dict, Tuple
from spacy.util import get_lang_class
from transformers import BertTokenizer
import numpy as np
import json


def load_json_lines(path: str) -> List[Dict]:
    """Load JSON lines.

    Parameters
    ----------
    path : str
        JSON file path

    Returns
    -------
    list
        List of dictionaries. One for each doccano document.

    """
    doccano_data = []
    with open(path) as f:
        for line in f:
            doccano_data.append(json.loads(line))

    return doccano_data


def extract_entities(
    doccano_document: Dict,
    include_indices: bool = False
) -> Dict[str, List]:
    """Extract entities from doccano document.

    Parameters
    ----------
    doccano_document : Dict
        Must contain 'text' and 'labels' key
    include_indices : bool, optional
        Contains indices if True, by default False

    Returns
    -------
    Dict[str, List]
        Dictionary with one key for every unique element.
        Values are list of entities.

    """
    text = doccano_document['text']
    labels = doccano_document['labels']
    unique_labels = {x[2] for x in labels}
    labels_dict: Dict = {k: [] for k in unique_labels}

    for label_item in labels:
        label_start, label_end, label = tuple(label_item)
        if include_indices:
            labels_dict[label].append(
                (label_start, label_end, text[label_start:label_end])
            )
        else:
            labels_dict[label].append(text[label_start:label_end])

    return labels_dict


def tokenize_with_indices(
    doccano_document: Dict,
    spacy_lang_class: str = 'da',
    bert_model: str = 'bert-base-multilingual-cased'
) -> List[Tuple[int, int, str]]:
    """Tokenize text and get indices for each tokenpiece.

    Parameters
    ----------
    doccano_document : Dict
        Must contain 'text' key
    spacy_lang_class : str, optional
        Two-letter language, by default 'da'
    bert_model : str, optional
        BERT model, by default 'bert-base-multilingual-cased'

    Returns
    -------
    List[Tuple[int, int, str]]
        Each tuple contains start and end index for token as well as the token

    """
    text = doccano_document['text']

    # Initiate spaCy sentencizer
    cls = get_lang_class(spacy_lang_class)
    sentencizer = cls()
    sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))

    # Initiate BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Sentencize text
    sentences = sentencizer(text).sents

    token_tuples = []

    # For each sentence
    for sentence_idx, sentence in enumerate(sentences):
        words, word_to_token_mapping = [], []

        # For each word in current sentence, tokenize word and store mapping
        for word in sentence:
            words.append(word)
            tokenpieces = tokenizer.tokenize(word.text)
            word_to_token_mapping.append((word, tokenpieces))

        # Map labels to token
        for word, tokens in word_to_token_mapping:
            # Computer tokens length
            tokens_len = [
                len(token) - 2 if token.startswith('##') else len(token)
                for token in tokens
            ]

            # Computer tokens indices
            token_start_idxs = word.idx + np.cumsum([0] + tokens_len[:-1])

            # Store start index, end index and token
            token_tuples.append(list(zip(
                token_start_idxs, token_start_idxs + tokens_len, tokens
            )))

    # Unnest
    token_tuples_unnested = [
        tokenpiece
        for word in token_tuples
        for tokenpiece in word
    ]

    return token_tuples_unnested
