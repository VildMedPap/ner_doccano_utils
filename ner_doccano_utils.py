"""
NER doccano utilites.

Collection of utility functions for during NER preprocessing.
"""

from typing import List, Dict, Tuple
from spacy.util import get_lang_class
from transformers import BertTokenizer
import numpy as np
import json
from seaborn import color_palette
from IPython.core.display import display, HTML


class Tokenizers:
    """Class holding spaCy sentencizer and BERT tokenizer."""

    def __init__(
        self,
        spacy_lang_class: str = 'da',
        bert_model: str = 'bert-base-multilingual-cased'
    ) -> None:
        """Initiate sentencizer and tokenizer.

        Parameters
        ----------
        spacy_lang_class : str, optional
            Two-letter language, by default 'da'
        bert_model : str, optional
            BERT model, by default 'bert-base-multilingual-cased'

        """
        # Initiate spaCy sentencizer
        lang_class = get_lang_class(spacy_lang_class)
        self.sentencizer = lang_class()
        self.sentencizer.add_pipe(self.sentencizer.create_pipe('sentencizer'))

        # Initiate BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)


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


def tokenize(
    doccano_document: Dict,
    tokenizers: Tokenizers
) -> List[Tuple[int, int, str]]:
    """Tokenize text and get indices for each tokenpiece.

    Parameters
    ----------
    doccano_document : Dict
        Must contain 'text' key
    tokenizers : Tokenizers
        Tokenizers instantiated with Tokenizers class (spaCy and BERT models)

    Returns
    -------
    List[Tuple[int, int, str]]
        Each tuple contains start and end index for token as well as the token

    """
    text = doccano_document['text']

    # Sentencize text
    sentences = tokenizers.sentencizer(text).sents

    token_tuples = []

    # For each sentence
    for sentence_idx, sentence in enumerate(sentences):
        words, word_to_token_mapping = [], []

        # For each word in current sentence, tokenize word and store mapping
        for word in sentence:
            words.append(word)
            tokenpieces = tokenizers.tokenizer.tokenize(word.text)
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


def distribute_labels(
    doccano_document: Dict,
    tokens: List[Tuple[int, int, str]]
) -> List[str]:
    """Distribute labels from doccano document to tokens.

    Parameters
    ----------
    doccano_document : Dict
        Must contain 'labels' key
    tokens : List[Tuple[int, int, str]]
        Each tuple contains start and end index for token as well as the token

    Returns
    -------
    List[str]
        List of labels. One for each token.

    """
    labels = doccano_document['labels']
    token_labels = []

    # For each token
    for token_start, token_end, token in tokens:

        # For each label
        token_label = ''
        for label_item in labels:
            label_start, label_end, label = tuple(label_item)

            if label_start <= token_start < label_end:
                token_label = label
        else:
            token_labels.append(token_label)

    return token_labels


def iob(labels: List[str]) -> List[str]:
    """Map labels to IOB schema.

    Parameters
    ----------
    labels : List[str]
        List of labels

    Returns
    -------
    List[str]
        List of labels in IOB format

    """
    labels = [
        'I-' + label.upper()
        if label != '' else 'O'
        for label in labels
    ]

    for i in range(len(labels) - 1):
        label = labels[i]
        next_label = labels[i + 1]

        if i == 0:
            candidate = label
            count = 0

        if candidate == label and i != 0:
            count += 1
        else:
            candidate = label
            count = 0

        if label == 'O':
            pass
        else:
            if next_label == candidate and count == 0:
                labels[i] = 'B' + labels[i][1:]

    return labels


def doccano_to_iob_tokens(
    doccano_document: Dict,
    tokenizers: Tokenizers
) -> List[Tuple[str, str]]:
    """Convert doccano document to tokens and iob labels.

    Parameters
    ----------
    doccano_document : Dict
        Must contain 'text' key
    tokenizers : Tokenizers
        Tokenizers instantiated with Tokenizers class (spaCy and BERT models)

    Returns
    -------
    List[Tuple[str, str]]
        Each tuples contains token and the corresponding IOB label

    """
    tokens = tokenize(doccano_document, tokenizers)
    labels = distribute_labels(doccano_document, tokens)
    iob_labels = iob(labels)

    tokenpieces = [x[2] for x in tokens]

    return list(zip(tokenpieces, iob_labels))


def create_color_dict(labels: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Create color dictionary.

    Parameters
    ----------
    labels : List[str]
        List of labels as strings

    Returns
    -------
    Dict[str, Tuple[int, int, int]]
        Dictionary with labels as keys and values as tuples of integers between
        0 and 255 (rgb scale values)
    """
    palette = color_palette(None, len(labels))

    color_dict = {}
    for lab, pal in zip(labels, palette):
        color_dict[lab] = (
            int(pal[0]*255),
            int(pal[1]*255),
            int(pal[2]*255),
        )

    return color_dict


def create_highlight_boxes(
    color_dict: Dict[str, Tuple[int, int, int]],
    a: float = 0.1,
    border_size: int = 1,
) -> Dict[str, str]:
    """Create highlight boxes.

    Parameters
    ----------
    color_dict : Dict[str, Tuple[int, int, int]]
        Dictionary with labels as keys and values as tuples of integers between
        0 and 255 (rgb scale values)
    a : float, optional
        Alpha value, transparency of highlight boxes , by default 0.1
    border_size : int, optional
        Border size of highlight boxes, by default 1

    Returns
    -------
    Dict[str, str]
        Dictionary with labels as keys and values as strings of html formatting

    """
    highlight_boxes = color_dict.copy()
    border_style = 'solid'

    for idx, (label, rgb) in enumerate(color_dict.items()):
        if idx > 9:
            border_style = 'dashed'

        r, g, b = rgb
        highlight_boxes[label] = f"""<span style="border: {border_size}px """ \
            f"""{border_style} rgb({r}, {g}, {b});""" \
            f"""background-color: rgba({r}, {g}, {b}, {a})">"""

    return highlight_boxes


def create_legend(highlight_boxes: Dict[str, str]) -> None:
    """Create legend.

    Parameters
    ----------
    highlight_boxes : Dict[str, str]
        Dictionary with labels as keys and values as strings of html formatting

    """
    text = '<b>LEGEND</b>:'
    space = '&nbsp;'
    for key, val in highlight_boxes.items():
        text += f'{space*4}{val}{space}{key}{space}</span>'

    div = f'''<div style="line-height:3;font-family:Verdana;''' \
          f'''font-size:13px">{text}</div>'''

    display(HTML(div))


def display_annotations(
    doccano_document: Dict,
    break_lines: bool = True
) -> None:
    """Display text with annotations.

    Parameters
    ----------
    doccano_document : Dict
        Must contain 'text' and 'labels' keys
    break_lines : bool, optional
        Whether or not to show linebreaks (\\n), by default True

    """
    labels = list({x[2] for x in doccano_document['labels']})
    color_dict = create_color_dict(labels)
    highlight_boxes = create_highlight_boxes(color_dict)

    text = doccano_document['text']
    for label in doccano_document['labels'][::-1]:
        start = label[0]
        end = label[1]
        text = text[:start] + highlight_boxes[label[2]] + \
            text[start:end] + '</span>' + text[end:]

    if break_lines:
        text = text.replace('\n', '<br/>')

    create_legend(highlight_boxes)
    display(HTML(
        '<div style="line-height: 2;font-family: Verdana; font-size: 15px; '
        'border: 1px solid grey; box-shadow: 0 0 5px rgba(0, 0, 0, 0.3), '
        f'inset 0 0 50px rgba(0, 0, 0, 0.025); padding: 10px">{text}</div>'
    ))


def split_doc_into_sentences(
    document,
    tokenizers: Tokenizers,
    MAX_LEN: int = 128
) -> List[List[str]]:
    """Split document into sentences of maximum length.

    Parameters
    ----------
    document : [type]
        String
    MAX_LEN : int, optional
        Maximum number of tokens in a sentence, by default 128
    tokenizers : Tokenizers
        Tokenizers instantiated with Tokenizers class (spaCy and BERT models)

    Returns
    -------
    List[List[str]]
        List of list of strings, which corresponds to one list of strings per
        sentence

    """
    def split_sentence(
        tokens: List[str],
        MAX_LEN: int = MAX_LEN
    ) -> List[List[str]]:
        """Split sentence into chunks of maximum length.

        Parameters
        ----------
        tokens : List[str]
            List of tokens
        MAX_LEN : int, optional
            Maximum number of tokens in a sentence, by default MAX_LEN

        Returns
        -------
        List[List[str]]
            List of list of strings, which corresponds to one list of strings
            per sentence

        """
        chunks = (len(tokens) + MAX_LEN - 1) // MAX_LEN
        return [tokens[i * MAX_LEN:(i + 1) * MAX_LEN] for i in range(chunks)]

    MAX_LEN = 128
    sentences = tokenizers.sentencizer(document).sents
    tokens_list = []
    for sentence in sentences:
        tokens = tokenizers.tokenizer.tokenize(sentence.text)

        if len(tokens) > MAX_LEN:
            chunks = split_sentence(tokens)
            [tokens_list.append(x) for x in chunks]
        else:
            tokens_list.append(tokens)

    return tokens_list
