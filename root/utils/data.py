"""
Loads JSONL datasets into memory. Includes a simple batching helper.
"""

import json


def batched(iterable, batch_size):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_sampels(data_path):
    """
    Loades whole dataset at once
    """
    with open(data_path, "r") as json_file:
        json_list = list(json_file)
        print("Data loaded.\n")

    questions_answers = {}
    for i, json_str in enumerate(json_list):
        questions_answers[i] = json.loads(json_str)

    return questions_answers
