import os, csv, shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class DatasetFormatter:
    def __init__(self, dataset_folder, output_folder):
        self.dataset_folder = dataset_folder
        self.output_folder = os.path.join(output_folder, "Images", "SanFranciscan")
        self.output_csv = os.path.join(output_folder, "Dataframes", "SanFranciscan.csv")
        self.load_filenames()
        self.fit_knn()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_filenames(self):
        self.dataset_paths = sorted(glob(os.path.join(self.dataset_folder, "**", "*.jpg"), recursive=True))
        self.dataset_info = [self.parse_filename(path) for path in self.dataset_paths]

    def fit_knn(self):
        utms = np.array([(info['utm_east'], info['utm_north']) for info in self.dataset_info]).astype(float)
        self.knn = NearestNeighbors(n_jobs=-1)
        self.knn.fit(utms)

    def parse_filename(self, path):
        parts = path.split("@")
        info = {
            'utm_east': float(parts[1]),
            'utm_north': float(parts[2]),
            'city_id': 'SanFranciscan',
            'northdeg': parts[9],
            'lat': parts[5],
            'lon': parts[6],
            'panoid': parts[7],
            'timestamp': parts[13],
        }
        return info

    def to_gsv_format(self):
            
        indices = self.knn.radius_neighbors([[info['utm_east'], info['utm_north']] for info in self.dataset_info], radius=25, return_distance=False)
        cluster_centers_indices = []

        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid'])
            writer.writeheader()

            place_id = 0
            for index in tqdm(indices):
                cluster_center_index = index[0]

                if cluster_center_index not in cluster_centers_indices:
                    cluster_centers_indices.append(cluster_center_index)

                    for sub_index in index:
                        original_path = self.dataset_paths[sub_index]
                        info = self.dataset_info[sub_index]
                        timestamp = info['timestamp']
                        year, month = timestamp[:4], timestamp[4:6]
                        writer.writerow({
                            'place_id': str(place_id).zfill(4),
                            'year': year,
                            'month': month,
                            'northdeg': info['northdeg'],
                            'city_id': info['city_id'],
                            'lat': info['lat'],
                            'lon': info['lon'],
                            'panoid': info['panoid'],
                        })
                        base_name = f"SanFranciscan_{str(place_id).zfill(7)}_{year}_{month}_{str(info['northdeg']).zfill(3)}_{float(info['lat'])}_{float(info['lon'])}_{info['panoid']}.jpg"
                        shutil.copyfile(original_path, os.path.join(self.output_folder, base_name))
                    place_id += 1

csv_generator = DatasetFormatter('/mnt/sda3/Projects/npr/datasets/sf_xl/small/train', '/mnt/sda3/Projects/npr/datasets/gsv_cities')
csv_generator.to_gsv_format()