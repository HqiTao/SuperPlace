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
        self.inliers_threshold = 100
        self.load_filenames()
        self.fit_database_knn()
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_filenames(self):
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.database_info = [self.parse_filename(path, class_name = "database") for path in self.database_paths]

    def fit_database_knn(self):
        database_utms = np.array([(info['utm_east'], info['utm_north']) for info in self.database_info]).astype(float)
        self.knn_database = NearestNeighbors(n_jobs=-1)
        self.knn_database.fit(database_utms)

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

        feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(image1)

        matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return len(m_kpts0)

    def process_database(self, index, place_id, queries_index, writer, database_record):

        matches_sum = {}
        num_queries = len(queries_index)
        for i, query_id in enumerate(queries_index):
            query_image_path = self.queries_paths[query_id]

            for sub_index in index:
                if sub_index not in database_record:
                    database_image_path = self.database_paths[sub_index]
                    num_inliers = self.image_matching(query_image_path, database_image_path)
                    if sub_index in matches_sum:
                        matches_sum[sub_index] += num_inliers
                    else:
                        matches_sum[sub_index] = num_inliers

        sorted_matches = sorted(matches_sum.items(), key=lambda x: x[1], reverse=True)

        for sub_index, num_inliers in sorted_matches[:10]:
            if (num_inliers/num_queries)> self.inliers_threshold:
                database_record.append(sub_index)
                original_path = self.database_paths[sub_index]
                info = self.database_info[sub_index]
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
    

    def process_queries(self, index, place_id, queries_index, writer, queries_record):

        matches_sum = {}
        for i, query_id in enumerate(queries_index):
            query_image_path = self.queries_paths[query_id]

            for sub_index in index:
                if self.queries_info[sub_index]['utm_east'] != self.queries_info[query_id]['utm_east'] and \
                    sub_index not in queries_record:
                    database_image_path = self.queries_paths[sub_index]
                    num_inliers = self.image_matching(query_image_path, database_image_path)
                    if sub_index in matches_sum:
                        matches_sum[sub_index] += num_inliers
                    else:
                        matches_sum[sub_index] = num_inliers

        sorted_matches = sorted(matches_sum.items(), key=lambda x: x[1], reverse=True)

        for sub_index, num_inliers in sorted_matches[:10]:
            if (num_inliers/2) > self.inliers_threshold:
                queries_index.append(sub_index)

        for sub_index in queries_index:
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



    def to_gsv_format(self):
        
        # cluster queries by utm
        queries_indices = self.knn_queries.radius_neighbors([[info['utm_east'], info['utm_north']] for info in self.queries_info], radius=10, return_distance=False)
        database_indices = self.knn_database.radius_neighbors([[info['utm_east'], info['utm_north']] for info in self.queries_info], radius=10, return_distance=False)

        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid'])
            writer.writeheader()

            place_id = 0
            refine_queries_indices = []
            queries_record = []
            for i, queries_path in enumerate(tqdm(self.queries_paths)):
                query_info = self.queries_info[i]
                if i not in queries_record and query_info['cameraid'] in ['00', '03', '06', '09']:
                    pose_query = [i, i+12]
                    gap_query = [i+1, i+2, i+13, i+14]
                    # queries as database
                    self.process_queries(queries_indices[i], place_id, pose_query, writer, queries_record)
                    queries_record.extend(pose_query)
                    queries_record.extend(gap_query)
                    refine_queries_indices.append(pose_query)
                    place_id += 1

            database_record = []
            for i, index in enumerate(tqdm(refine_queries_indices)): # i is place_id
                if len(database_indices[index[0]]) != 0:
                    self.process_database(database_indices[index[0]], i, index, writer, database_record)


csv_generator = DatasetFormatter('/media/hello/data1/binux/datasets/pitts30k/images/train', 
                                 '/media/hello/data1/binux/datasets/gsv_cities')
csv_generator.to_gsv_format()