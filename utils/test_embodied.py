import faiss, os
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils import visualizations

import numpy as np
import matplotlib.pyplot as plt
 
def weighted_average_feature(features, weights=[0.1, 0.8, 0.1]):
    weighted_feature = np.zeros_like(features[0])
    for feature, weight in zip(features, weights):
        weighted_feature += weight * feature
    return weighted_feature


def test(args, eval_ds, model):
    """Compute features of the given dataset and compute the recalls."""
    
    # if args.efficient_ram_testing:
        # return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
        database_features_dir = os.path.join(eval_ds.dataset_folder, "database_features.npy")
        queries_features_dir = os.path.join(eval_ds.dataset_folder, "queries_features.npy")

        if os.path.isfile(database_features_dir) == 1:
            database_features = np.load(database_features_dir)
        else: 
            for images, indices in tqdm(database_dataloader, ncols=100):
                features = model(images.to("cuda"))
                features = features.cpu().numpy()
                all_features[indices.numpy(), :] = features
            
            database_features = all_features[:eval_ds.database_num]
            np.save(database_features_dir, database_features)
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = args.infer_batch_size
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        
        if os.path.isfile(queries_features_dir) == 1:
            queries_features = np.load(queries_features_dir)
        else: 
            for inputs, indices in tqdm(queries_dataloader, ncols=100):
                features = model(inputs.to("cuda"))
                features = features.cpu().numpy()
                # if pca is not None:
                #     features = pca.transform(features)
                
                all_features[indices.numpy(), :] = features
            
            queries_features = all_features[eval_ds.database_num:]
            np.save(queries_features_dir, queries_features)

    
    # queries_features = all_features[eval_ds.database_num:]
    # database_features = all_features[:eval_ds.database_num]

    # similarity_matrix = np.dot(queries_features, database_features.T)

    # plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Self-similarity Matrix')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Sample Index')
    # plt.savefig(f"ss.png")
        
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del all_features

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))

    ''' 
    L2(x - (y1 + y2 + y3) / 3)
    '''
    # rank_n = 100
    # new_predictions = []
    # mixup_database_features = database_features.copy()

    # for idx in tqdm(range(mixup_database_features.shape[0])):
    #     if idx > 0 and idx < eval_ds.database_num - 1:
    #         mixup_database_features[idx] = weighted_average_feature([database_features[idx-1],
    #                                                                  database_features[idx],
    #                                                                  database_features[idx+1]])

    # for query_idx in range(predictions.shape[0]):
    #     prediction = predictions[query_idx]
    #     query_feature = queries_features[query_idx]

    #     distances = np.linalg.norm(mixup_database_features[prediction[:rank_n]] - query_feature, axis=1)
    #     ranked_indices = np.argsort(distances)

    #     ranked_prediction = prediction[:rank_n][ranked_indices]

    #     unranked_predictions = prediction[rank_n:]
    #     new_prediction = np.concatenate((ranked_prediction, unranked_predictions))

    #     new_predictions.append(new_prediction)

    # predictions = np.array(new_predictions)

    ''' 
    L2(max (y1 + y2 + y3) - y)
    '''
    # rank_n = 100
    # new_predictions = []
    # mixup_database_features = database_features.copy()

    # for idx in tqdm(range(mixup_database_features.shape[0])):
    #     if idx > 0 and idx < eval_ds.database_num - 1:
    #         mixup_database_features[idx] = weighted_average_feature([database_features[idx-1],
    #                                                                  database_features[idx],
    #                                                                  database_features[idx+1]])

    # for query_idx in range(predictions.shape[0]):
    #     prediction = predictions[query_idx]
    #     query_feature = queries_features[query_idx]
    #     mixup_query_feature = np.max(mixup_database_features[prediction[:5]], axis=0)

    #     distances_mixup_database = np.linalg.norm(mixup_database_features[prediction[:rank_n]] - query_feature, axis=1)
    #     distances_mixup_query = np.linalg.norm(database_features[prediction[:rank_n]] - mixup_query_feature, axis=1)
    #     ranked_indices = np.argsort(distances_mixup_query)

    #     ranked_prediction = prediction[:rank_n][ranked_indices]

    #     unranked_predictions = prediction[rank_n:]
    #     new_prediction = np.concatenate((ranked_prediction, unranked_predictions))

    #     new_predictions.append(new_prediction)

    # predictions = np.array(new_predictions)

    '''
    [L2(x - y1) + L2(x - y2) + L2(x - y3)] / 3
    '''

    # rank_n = 100
    # weights = np.array([0.2, 0.6, 0.2])
    # new_predictions = []

    # for query_idx in range(predictions.shape[0]):
    #     prediction = predictions[query_idx]
    #     query_feature = queries_features[query_idx]

    #     distances = np.zeros(rank_n)
    #     for idx in range(prediction.shape[0]):
    #         database_idx = prediction[idx]
    #         if database_idx > 0 and database_idx < eval_ds.database_num - 1:
    #             distance = np.linalg.norm(database_features[database_idx-1: database_idx+2] - query_feature, axis=1)
    #             distance = np.sum(distance * weights)
    #         else:
    #             distance = np.linalg.norm(database_features[database_idx] - query_feature)
    #         distances[idx] = distance
    #     ranked_indices = np.argsort(distances)

    #     ranked_prediction = prediction[:rank_n][ranked_indices]

    #     unranked_predictions = prediction[rank_n:]
    #     new_prediction = np.concatenate((ranked_prediction, unranked_predictions))

    #     new_predictions.append(new_prediction)

    # predictions = np.array(new_predictions)


    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)


    return recalls, recalls_str