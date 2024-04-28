import faiss
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

def test(args, eval_ds, model):

    model = model.eval()
    feature_maps = []
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=1, pin_memory=True)
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            B, C, H, W = inputs.shape
            feature_map, cls_token = model(inputs.to("cuda"))
            feature_maps.append(feature_map)
        patch_h, patch_w = H//14, W//14
        feature_maps = torch.cat(feature_maps, dim = 0)

        feature_maps = feature_maps.reshape(-1, 768).cpu()

        pca = PCA(n_components=3)
        pca.fit(feature_maps)
        pca_features = pca.transform(feature_maps)

        components = pca.components_

        for i in range(3):
            component = components[i]
            contributions = np.abs(component)
            most_contributive_channels = contributions.argsort()[-10:][::-1]

            print(f"Principal Component {i+1}: Top contributing channels: {most_contributive_channels}")
            print(f"Contributions: {contributions[most_contributive_channels]}")

        choice_dim = 2
        pca_features[:, choice_dim] = (pca_features[:, choice_dim] - pca_features[:, choice_dim].min()) / \
                     (pca_features[:, choice_dim].max() - pca_features[:, choice_dim].min())

        for i in range(7):
            plt.subplot(3, 3, i+1)
            plt.imshow(pca_features[i*patch_h*patch_w : (i+1)*patch_h*patch_w, choice_dim].reshape(patch_h, patch_w))
            plt.axis('off')

        plt.savefig(f"pca_{choice_dim}.png")

        pca_features_bg = pca_features[:, 0] > 0.35
        pca_features_fg = ~pca_features_bg

        for i in range(7):
            plt.subplot(3, 3, i+1)
            plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
            plt.axis('off')
        plt.savefig(f"1st_{choice_dim}.png")


        pca.fit(feature_maps[pca_features_bg]) 
        pca_features_left = pca.transform(feature_maps[pca_features_bg])

        for i in range(3):
            # min_max scaling
            pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

        pca_features_rgb = pca_features.copy()
        # for black background
        pca_features_rgb[pca_features_fg] = 0
        # new scaled foreground features
        pca_features_rgb[pca_features_bg] = pca_features_left

        # reshaping to numpy image format
        pca_features_rgb = pca_features_rgb.reshape(7, patch_h, patch_w, 3)
        for i in range(7):
            plt.subplot(3, 3, i+1)
            plt.imshow(pca_features_rgb[i])

        plt.axis('off')
        plt.savefig(f"rgb_{choice_dim}.png")