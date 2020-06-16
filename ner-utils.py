"""
NER utilites.

Collection of utility functions for during NER preprocsseing.
Especially when data comes as doccano data.
"""

from typing import List, Dict
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
