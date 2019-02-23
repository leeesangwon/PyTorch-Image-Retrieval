# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader
from data_loader import test_data_generator

import numpy as np


def retrieve(model_list, queries, db, img_size_list, batch_size, query_expansion=False):
    assert len(model_list) == len(img_size_list), "model_list and img_size_list should have same length"

    query_paths = queries
    reference_paths = db

    query_img_dataset = test_data_generator(queries, img_size=img_size_list[0], flip=False)
    reference_img_dataset = test_data_generator(db, img_size=img_size_list[0], flip=False)

    query_loader = DataLoader(query_img_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=True)
    reference_loader = DataLoader(reference_img_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                  pin_memory=True)

    sim_matrix_list = []

    for model in model_list:
        # inference
        model.eval()
        model.cuda()

        query_paths, query_vecs = batch_process(model, query_loader)
        reference_paths, reference_vecs = batch_process(model, reference_loader)

        assert query_paths == queries and reference_paths == db, "order of paths should be same"

        query_vecs, reference_vecs = db_augmentation(query_vecs, reference_vecs, top_k=10)
        query_vecs, reference_vecs = db_qe(query_vecs, reference_vecs, top_k=5)     # Round 1

        sim_matrix_list.append(calculate_sim_matrix(query_vecs, reference_vecs))

    sim_matrix_cat = np.array(sim_matrix_list)
    avg_sim_matrix = np.mean(sim_matrix_cat, axis=0)

    # 여기서부터는 안건드려도 됨.
    indices = np.argsort(avg_sim_matrix, axis=1)
    indices = np.flip(indices, axis=1)

    retrieval_results = {}

    for (i, query) in enumerate(query_paths):
        query = query.split('/')[-1].split('.')[0]
        ranked_list = [reference_paths[k].split('/')[-1].split('.')[0] for k in indices[i]]
        ranked_list = ranked_list[:1000]

        retrieval_results[query] = ranked_list
    print('done')

    return retrieval_results


def db_augmentation(query_vecs, reference_vecs, top_k=10):
    weights = np.logspace(0, -2., top_k+1)
    # Query augmentation
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))

    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref = reference_vecs[indices[:, :top_k+1], :]
    reference_vecs = np.tensordot(weights, top_k_ref, axes=(0, 1))

    return query_vecs, reference_vecs


def db_qe(query_vecs, reference_vecs, top_k=5):
    # Query augmentation
    sim_mat = calculate_sim_matrix(query_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref_mean = np.mean(reference_vecs[indices[:, :top_k], :], axis=1)
    query_vecs = np.concatenate([query_vecs, top_k_ref_mean], axis=1)

    # Reference augmentation
    sim_mat = calculate_sim_matrix(reference_vecs, reference_vecs)
    indices = np.argsort(-sim_mat, axis=1)

    top_k_ref_mean = np.mean(reference_vecs[indices[:, 1:top_k+1], :], axis=1)
    reference_vecs = np.concatenate([reference_vecs, top_k_ref_mean], axis=1)

    return query_vecs, reference_vecs


def calculate_sim_matrix(query_vecs, reference_vecs):
    query_vecs, reference_vecs = postprocess(query_vecs, reference_vecs)
    return np.dot(query_vecs, reference_vecs.T)


def batch_process(model, loader):
    feature_vecs = []
    img_paths = []
    for data in loader:
        paths, inputs = data
        feature_vec = _get_feature(model, inputs.cuda())
        feature_vec = feature_vec.detach().cpu().numpy()  # (batch_size, channels)
        for i in range(feature_vec.shape[0]):
            feature_vecs.append(feature_vec[i])
        img_paths = img_paths + paths

    return img_paths, np.asarray(feature_vecs)


def _get_features_from(model, x, feature_names):
    features = {}

    def save_feature(name):
        def hook(m, i, o):
            features[name] = o.data

        return hook

    for name, module in model.named_modules():
        _name = name.split('.')[-1]
        if _name in feature_names:
            module.register_forward_hook(save_feature(_name))

    model(x)

    return features


def _get_feature(model, x):
    model_name = model.__class__.__name__

    if model_name == 'DenseNet':
        features = _get_features_from(model, x, ['classifier'])
        feature = features['classifier']
    elif model_name == 'ResNet':
        features = _get_features_from(model, x, ['fc'])
        feature = features['fc']
    elif model_name == 'EmbeddingNetwork':
        feature = model(x)
    elif model_name == 'EmbeddingNetwork_Dropout':
        feature = model(x)
    else:
        raise ValueError("Invalid model name: {}".format(model_name))

    return feature


def postprocess(query_vecs, reference_vecs):
    # centerize
    query_vecs, reference_vecs = _centerize(query_vecs, reference_vecs)

    # l2 normalization
    query_vecs = _l2_normalize(query_vecs)
    reference_vecs = _l2_normalize(reference_vecs)

    return query_vecs, reference_vecs


def _centerize(v1, v2):
    concat = np.concatenate([v1, v2], axis=0)
    center = np.mean(concat, axis=0)
    return v1-center, v2-center


def _l2_normalize(v):
    norm = np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
    if np.any(norm == 0):
        return v
    return v / norm
