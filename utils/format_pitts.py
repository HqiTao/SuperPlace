import os, csv, shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class DatasetFormatter:
    def __init__(self, dataset_folder, output_folder):
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(self.dataset_folder, "database")
        self.queries_folder = os.path.join(self.dataset_folder, "queries")
        self.output_folder = os.path.join(output_folder, "Images", "Pittsburgh")
        self.output_csv = os.path.join(output_folder, "Dataframes", "Pittsburgh.csv")
        self.load_filenames()
        self.fit_queries_knn()
        self.fit_database_knn()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_filenames(self):
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        self.database_info = [self.parse_filename(path) for path in self.database_paths]
        self.queries_info = [self.parse_filename(path) for path in self.queries_paths]

    def fit_database_knn(self):
        database_utms = np.array([(info['utm_east'], info['utm_north']) for info in self.database_info]).astype(float)
        self.knn_database = NearestNeighbors(n_jobs=-1)
        self.knn_database.fit(database_utms)

    def fit_queries_knn(self):
        queries_utms = np.array([(info['utm_east'], info['utm_north']) for info in self.queries_info]).astype(float)
        self.knn_queries = NearestNeighbors(n_jobs=-1)
        self.knn_queries.fit(queries_utms)

    def parse_filename(self, path):
        parts = path.split("@")
        info = {
            'utm_east': float(parts[1]),
            'utm_north': float(parts[2]),
            'city_id': 'Pittsburgh',
            'northdeg': parts[9],
            'lat': parts[5],
            'lon': parts[6],
            'panoid': parts[7],
            'timestamp': parts[13],
        }
        return info

    def cluster_queries(self):
        pass

        

    def to_gsv_format(self):
        pass

        

        # with open(self.output_csv, mode='w', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid'])
        #     writer.writeheader()

        #     place_id = 0
        #     for index in tqdm(indices):
        #         cluster_center_index = index[0]

        #         if cluster_center_index not in cluster_centers_indices:
        #             cluster_centers_indices.append(cluster_center_index)

        #             for sub_index in index:
        #                 original_path = self.dataset_paths[sub_index]
        #                 info = self.dataset_info[sub_index]
        #                 timestamp = info['timestamp']
        #                 year, month = timestamp[:4], timestamp[4:6]
        #                 writer.writerow({
        #                     'place_id': str(place_id).zfill(4),
        #                     'year': year,
        #                     'month': month,
        #                     'northdeg': info['northdeg'],
        #                     'city_id': info['city_id'],
        #                     'lat': info['lat'],
        #                     'lon': info['lon'],
        #                     'panoid': info['panoid'],
        #                 })
        #                 base_name = f"SanFranciscan_{str(place_id).zfill(7)}_{year}_{month}_{str(info['northdeg']).zfill(3)}_{float(info['lat'])}_{float(info['lon'])}_{info['panoid']}.jpg"
        #                 shutil.copyfile(original_path, os.path.join(self.output_folder, base_name))
        #             place_id += 1

csv_generator = DatasetFormatter('/media/hello/data1/binux/datasets/pitts30k/images/train', '/media/hello/data1/binux/datasets/gsv_cities')
csv_generator.to_gsv_format()