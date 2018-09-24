from util.embeddings import read_embeddings_file
from util.util import save_json


def process(glove_path, save_path):
    embeddings_index = read_embeddings_file(glove_path)
    save_json(save_path, list(set(embeddings_index.keys())), format_json=False)


if __name__ == "__main__":
    process('E:/data_sets/glove/glove.840B.300d.txt', '../data/embeddings/vocab.json')
