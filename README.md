# NER utilities

## Example data

### `danish.json`

```python
[
    {
        'text': 'Og så skyder han. Nej, han venter. Jaaaa, Michael Laudrup. Det er genialt, dét der!',
        'labels': [[42, 57, 'name']]
    },
    {
        'text': 'Der er ikke fejet noget ind under gulvtæppet, sagde Statsminister Poul Schlüter fra Folketingets talerstol 25. april 1989.',
        'labels': [[52, 65, 'title'], [66, 79, 'name'], [107, 121, 'date']]
    },
    {
        'text': 'Lige ud af Næstved - ikke af betonen. Det er der, jeg er født og vokset op!',
        'labels': [[11, 18, 'city']]
    }
]
```

## Utilities

### `load_json_lines()`

```python
doccano_data = load_json_lines('examples/danish.json')
```

```python
len(doccano_data)

output:
3
```

```python
type(doccano_data)

output:
list
```

```python
doccano_data[1]

output:
{'text': 'Der er ikke fejet noget ind under gulvtæppet, sagde Statsminister Poul Schlüter fra Folketingets talerstol 25. april 1989.',
 'labels': [[52, 65, 'title'], [66, 79, 'name'], [107, 121, 'date']]}
```

### `extract_entities()`

```python
extract_entities(doccano_data[1])

output:
{'title': ['Statsminister'],
 'name': ['Poul Schlüter'],
 'date': ['25. april 1989']}
```

```python
extract_entities(doccano_data[1], include_indices=True)

output:
{'title': [(52, 65, 'Statsminister')],
 'name': [(66, 79, 'Poul Schlüter')],
 'date': [(107, 121, '25. april 1989')]}
```