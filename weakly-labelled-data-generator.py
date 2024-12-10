import numpy as np
import random
from itertools import chain
from tqdm import tqdm
import ray
from flashtext import KeywordProcessor
import spacy
import multiprocessing

random.seed(0)
np.random.seed(seed=0)

sampling_rate_no_weak_labels = 0.1

nlp_tokenizer = spacy.blank('en')

with open('data/chem_dict.txt', 'r') as f:
    dict_chem = [x.strip() for x in f if x.strip() != ""]

with open('data/disease_dict.txt', 'r') as f:
    dict_disease = [x.strip() for x in f if x.strip() != ""]

print(len(dict_chem))
print(len(dict_disease))

entity_to_type = {}
for entity in dict_chem:
    entity_to_type[entity] = 'Chemical'
for entity in dict_disease:
    entity_to_type[entity] = 'Disease'


@ray.remote
def process_chunk(text_lines, entity_to_type):
    labeled_lines = []
    unlabeled_lines = []
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for entity, entity_type in entity_to_type.items():
        keyword_processor.add_keyword(entity, entity_type)
    for line in text_lines:
        doc = nlp_tokenizer(line)
        tokens = [token.text for token in doc]
        labels = ['O'] * len(tokens)
        token_positions = [(token.idx, token.idx + len(token)) for token in doc]
        matches = keyword_processor.extract_keywords(line, span_info=True)
        for match in matches:
            matched_entity_type = match[0]
            start_pos, end_pos = match[1], match[2]
            for i, (token_start, token_end) in enumerate(token_positions):
                if token_end <= start_pos:
                    continue
                if token_start >= end_pos:
                    break
                if labels[i] == 'O':
                    if token_start == start_pos:
                        labels[i] = 'B-' + matched_entity_type
                    else:
                        labels[i] = 'I-' + matched_entity_type
        if all(label == 'O' for label in labels):
            unlabeled_lines.append(line)
        else:
            labeled_lines.append([tokens, labels])
    return labeled_lines, unlabeled_lines


with open('data/cleaned_unlabeled_sentence_data.txt', 'r') as input_file:
    all_sentences = input_file.readlines()

num_cpus = multiprocessing.cpu_count()
num_chunks = num_cpus * 2
chunk_size = len(all_sentences) // num_chunks + 1

chunks = [all_sentences[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

futures = [process_chunk.remote(chunk, entity_to_type) for chunk in chunks]

all_processed_data = []
for result in tqdm(ray.get(futures), total=len(futures)):
    all_processed_data.append(result)

labeled_lines = list(chain.from_iterable([x[0] for x in all_processed_data]))
unlabeled_lines = list(chain.from_iterable([x[1] for x in all_processed_data]))

with open('data/weakly_labeled_data.txt', 'w') as output_file:
    for tokens, labels in labeled_lines:
        for word, label in zip(tokens, labels):
            output_file.write(f"{word}\t{label}\n")
        output_file.write("\n")

print(
    "Processing completed. Labeled data saved to 'weakly_labeled_data.txt' and unlabeled data saved to 'unlabeled_data.txt'.")
