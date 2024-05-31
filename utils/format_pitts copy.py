import os, csv, shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

class DatasetFormatter:
    def __init__(self, dataset_folder, output_folder):
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(self.dataset_folder, "database")
        self.queries_folder = os.path.join(self.dataset_folder, "queries")
        self.output_folder = os.path.join(output_folder, "Images", "Pittsburgh")
        self.output_csv = os.path.join(output_folder, "Dataframes", "Pittsburgh.csv")
        self.inliers_threshold = 224
        self.queries_record = []
        self.database_record = []
        self.load_filenames()
        self.fit_database_knn()
        self.fit_queries_knn()
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    

    def load_filenames(self):
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        self.database_info = [self.parse_filename(path, class_name = "database") for path in self.database_paths]
        self.queries_info = [self.parse_filename(path) for path in self.queries_paths]


    def fit_database_knn(self):
        database_utms = np.array([(info['utm_east'], info['utm_north']) for info in self.database_info]).astype(float)
        self.knn_database = NearestNeighbors(n_jobs=-1)
        self.knn_database.fit(database_utms)


    def fit_queries_knn(self):
        queries_utms = np.array([(info['utm_east'], info['utm_north']) for info in self.queries_info]).astype(float)
        self.knn_queries = NearestNeighbors(n_jobs=-1)
        self.knn_queries.fit(queries_utms)


    def parse_filename(self, path, class_name = "queries"):
        parts = path.split("@")
        info = {
            'utm_east': float(parts[1]),
            'utm_north': float(parts[2]),
            'city_id': 'Pittsburgh',
            'northdeg': '0',
            'lat': parts[5],
            'lon': parts[6],
            'panoid': class_name + parts[7],
            'cameraid': parts[8],
            'timestamp': '199607', # unused
        }
        return info
    

    def image_matching(self, img0_path, img1_path):

        image0 = load_image(img0_path).cuda()
        image1 = load_image(img1_path).cuda()

        feats0 = self.extractor.extract(image0, resize = None)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(image1, resize = None)

        matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return len(m_kpts0)

    def process_image_cvs(self, writer, index, place_id, class_name = "queries"):

        for sub_index in index:
            if class_name == "database":
                original_path = self.database_paths[sub_index]
                info = self.database_info[sub_index]
            else:
                original_path = self.queries_paths[sub_index]
                info = self.queries_info[sub_index]
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
                'panoid': info['panoid'] + info['cameraid'],
            })
            base_name = f"Pittsburgh_{str(place_id).zfill(7)}_{year}_{month}_{str(info['northdeg']).zfill(3)}_{float(info['lat'])}_{float(info['lon'])}_{info['panoid'] + info['cameraid']}.jpg"
            shutil.copyfile(original_path, os.path.join(self.output_folder, base_name))


    def process_database(self, candidate_index, queries_index, database_record):

        refine_database_index = []
        matches_sum = {}

        query_image_path = self.queries_paths[queries_index[0]]

        for sub_index in candidate_index:
            if sub_index not in database_record:
                database_image_path = self.database_paths[sub_index]
                num_inliers = self.image_matching(query_image_path, database_image_path)
                matches_sum[sub_index] = num_inliers

        sorted_matches = sorted(matches_sum.items(), key=lambda x: x[1], reverse=True)

        for sub_index, num_inliers in sorted_matches[:3]:
            if num_inliers > self.inliers_threshold:
                database_record.append(sub_index)
                refine_database_index.append(sub_index)

        return refine_database_index
    

    def process_queries(self, candidate_index, queries_index, queries_record):

        matches_sum = {}
        query_image_path = self.queries_paths[queries_index[0]]

        for sub_index in candidate_index:
            if self.queries_info[sub_index]['utm_east'] != self.queries_info[queries_index[0]]['utm_east'] and \
                sub_index not in queries_record:
                candidate_image_path = self.queries_paths[sub_index]
                num_inliers = self.image_matching(query_image_path, candidate_image_path)
                matches_sum[sub_index] = num_inliers

        sorted_matches = sorted(matches_sum.items(), key=lambda x: x[1], reverse=True)

        for sub_index, num_inliers in sorted_matches[:3]:
            if num_inliers > self.inliers_threshold:
                queries_index.append(sub_index)


    def to_gsv_format(self):
        
        # cluster queries by utm
        queries_indices = self.knn_queries.radius_neighbors([[info['utm_east'], info['utm_north']] for info in self.queries_info], radius=25, return_distance=False)
        database_indices = self.knn_database.radius_neighbors([[info['utm_east'], info['utm_north']] for info in self.queries_info], radius=25, return_distance=False)

        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid'])
            writer.writeheader()

            place_id = 0
            refine_queries_indices = []

            for i, queries_path in enumerate(tqdm(self.queries_paths)):
                query_info = self.queries_info[i]
                if i not in self.queries_record and query_info['cameraid'] in ['00', '03', '06', '09'] and len(database_indices[i]) != 0:
                    query_index = [i, i+12]
                    gap_query = [i+1, i+2, i+13, i+14]
                    # queries as database
                    self.process_queries(queries_indices[i], query_index, self.queries_record)
                    self.queries_record.extend(query_index)
                    self.queries_record.extend(gap_query)
                    refine_queries_indices.append(query_index)

            for i, query_index in enumerate(tqdm(refine_queries_indices)): # i is place_id
                if len(database_indices[query_index[0]]) != 0:
                    refine_database_index = self.process_database(database_indices[query_index[0]], query_index, self.database_record)
                    if len(refine_database_index) >= 2:
                        self.process_image_cvs(writer, refine_database_index, place_id, class_name="database")
                        self.process_image_cvs(writer, query_index, place_id, class_name="queries")
                        place_id += 1


csv_generator = DatasetFormatter('/mnt/sda3/Projects/npr/datasets/pitts30k/images/train', 
                                 '/mnt/sda3/Projects/npr/datasets/gsv_cities')
csv_generator.to_gsv_format()