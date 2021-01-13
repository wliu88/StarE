from functools import partial
from typing import List, Union, Dict, Callable
import numpy as np


def get_alternative_graph_repr(raw: Union[List[List[int]], np.ndarray], config: dict) \
        -> Dict[str, np.ndarray]:
    """
    Decisions:

        Quals are represented differently here, i.e., more as a coo matrix
        s1 p1 o1 qr1 qe1 qr2 qe2    [edge index column 0]
        s2 p2 o2 qr3 qe3            [edge index column 1]

        edge index:
        [ [s1, s2],
          [o1, o2] ]

        edge type:
        [ p1, p2 ]

        quals will looks like
        [ [qr1, qr2, qr3],
          [qe1, qr2, qe3],
          [0  , 0  , 1  ]       <- obtained from the edge index columns

    :param raw: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]]
        (already have a max qualifier length padded data)
    :param config: the config dict
    :return: output dict
    """
    has_qualifiers: bool = config['STATEMENT_LEN'] != 3
    try:
        nr = config['NUM_RELATIONS']
    except KeyError:
        raise AssertionError("Function called too soon. Num relations not found.")

    edge_index, edge_type = np.zeros((2, len(raw) * 2), dtype='int32'), np.zeros((len(raw) * 2), dtype='int32')
    # qual_rel = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype='int32')
    # qual_ent = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype='int32')
    qualifier_rel = []
    qualifier_ent = []
    qualifier_edge = []

    # Add actual data
    for i, data in enumerate(raw):
        edge_index[:, i] = [data[0], data[2]]
        edge_type[i] = data[1]

        # @TODO: add qualifiers
        if has_qualifiers:
            qual_rel = np.array(data[3::2])
            qual_ent = np.array(data[4::2])
            non_zero_rels = qual_rel[np.nonzero(qual_rel)]
            non_zero_ents = qual_ent[np.nonzero(qual_ent)]
            for j in range(non_zero_ents.shape[0]):
                qualifier_rel.append(non_zero_rels[j])
                qualifier_ent.append(non_zero_ents[j])
                qualifier_edge.append(i)

    quals = np.stack((qualifier_rel, qualifier_ent, qualifier_edge), axis=0)
    num_triples = len(raw)

    # Add inverses
    edge_index[1, len(raw):] = edge_index[0, :len(raw)]
    edge_index[0, len(raw):] = edge_index[1, :len(raw)]
    edge_type[len(raw):] = edge_type[:len(raw)] + nr

    if has_qualifiers:
        full_quals = np.hstack((quals, quals))
        full_quals[2, quals.shape[1]:] = quals[2, :quals.shape[1]]  # TODO: might need to + num_triples

        return {'edge_index': edge_index,
                'edge_type': edge_type,
                'quals': full_quals}
    else:
        return {'edge_index': edge_index,
                'edge_type': edge_type}


if __name__ == "__main__":
    config = dict()
    config['STATEMENT_LEN'] = 15
    config['NUM_RELATIONS'] = 10

    instances = [[1, 1, 2, 2, 3, 3, 4], [5, 4, 6, 5, 7, 6, 8]]

    g = get_alternative_graph_repr(instances, config)

    print(g["edge_index"])
    print(g["edge_type"])
    print(g["quals"])