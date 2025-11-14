import unittest
from pathlib import Path

import animl


@unittest.skip
def reid_test():
    manifest_path = Path.cwd() / 'examples' / 'Jaguar'
    miew_path = Path.cwd() / 'models/miewid_v3.onnx'
    manifest = animl.build_file_manifest(manifest_path)

    miew = animl.load_miew(miew_path)
    embeddings = animl.extract_miew_embeddings(miew,
                                               manifest,
                                               file_col="filepath")

    print(embeddings.shape)

    e2 = animl.compute_distance_matrix(embeddings, embeddings, metric='euclidean')
    cos = animl.compute_distance_matrix(embeddings, embeddings, metric='cosine')


reid_test()
