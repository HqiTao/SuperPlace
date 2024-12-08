# Baidu Mall dataset
"""
    Download from: https://www.dropbox.com/s/4mksiwkxb7t4a8a/IDL_dataset_cvpr17_3852.zip
    jar xf ./IDL_dataset_cvpr17_3852.zip
"""

# %%
import os
import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from natsort import natsorted
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from scipy.spatial.transform import Rotation

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

# %%
def get_cop_pose(file):
    """
    Takes in input of .camera file for baidu and outputs the cop numpy array [x y z] and 3x3 rotation matrix
    """
    with open(file) as f:
        lines = f.readlines()
        xyz_cop_line = lines[-2]
        # print(cop_line)
        xyz_cop = np.fromstring(xyz_cop_line, dtype=float, sep=' ')    

        r1 = np.fromstring(lines[4], dtype=float, sep=' ')
        r2 = np.fromstring(lines[5], dtype=float, sep=' ')
        r3 = np.fromstring(lines[6], dtype=float, sep=' ')
        r =  Rotation.from_matrix(np.array([r1,r2,r3]))
        # print(R)

        R_euler = r.as_euler('zyx', degrees=True)

    return xyz_cop,R_euler

def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

class BaiduDataset(data.Dataset):
    def __init__(self, args, dataset_name="baidu_datasets", short_size=736,\
                  use_ang_positives=False, dist_thresh = 10, ang_thresh=20,):
        self.dataset_name = dataset_name
        self.datasets_folder = args.datasets_folder

        self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"training_images_undistort")))
        self.db_gt_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"training_gt")))
        self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"query_images_undistort")))
        self.q_gt_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"query_gt")))

        self.db_abs_paths = []
        self.q_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"training_images_undistort",p))

        for q in self.q_paths:
            self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"query_images_undistort",q))

        self.ang_thresh = ang_thresh
        self.dist_thresh = dist_thresh

        self.short_size = short_size

        self.db_num = len(self.db_paths)
        self.q_num = len(self.q_paths)

        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.transform = transforms.Compose([
            transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.BILINEAR) if args.resize_test_imgs else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std'])
        ])

        #form pose array from db_gt .camera files
        self.db_gt_arr = np.zeros((self.db_num,3)) #for xyz
        self.db_gt_arr_euler = np.zeros((self.db_num,3)) #for euler angles

        for idx,db_gt_file_rel in enumerate(self.db_gt_paths):

            db_gt_file = os.path.join(self.datasets_folder,self.dataset_name,"training_gt",db_gt_file_rel)

            with open(db_gt_file) as f:
                cop_pose,cop_R = get_cop_pose(db_gt_file)
                
            self.db_gt_arr[idx,:] = cop_pose
            self.db_gt_arr_euler[idx,:] = cop_R

        #form pose array from q_gt .camera files
        self.q_gt_arr = np.zeros((self.q_num,3)) #for xyz
        self.q_gt_arr_euler = np.zeros((self.q_num,3)) #for euler angles

        for idx,q_gt_file_rel in enumerate(self.q_gt_paths):

            q_gt_file = os.path.join(self.datasets_folder,self.dataset_name,"query_gt",q_gt_file_rel)

            with open(q_gt_file) as f:
                cop_pose,cop_R = get_cop_pose(q_gt_file)
            
            self.q_gt_arr[idx,:] = cop_pose
            self.q_gt_arr_euler[idx,:] = cop_R

        if use_ang_positives:
            # Find soft_positives_per_query, which are within val_positive_dist_threshold and ang_threshold
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_dist_positives_per_query = knn.radius_neighbors(self.q_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)

            #also apply the angular distance threshold
            self.soft_positives_per_query = []

            for i in range(len(self.q_gt_arr)):                 #iterate over all q_gt_array
                self.ang_dist = []
                for j in range(len(self.soft_dist_positives_per_query[i])): #iterate over all positive queries
                    # print(self.q_gt_arr - self.db_gt_arr[self.soft_positives_per_query[i][j]])
                    ang_diff = np.mean(np.abs(self.q_gt_arr_euler[i] - self.db_gt_arr_euler[self.soft_dist_positives_per_query[i][j]]))
                    if ang_diff<self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_query[i][j])
                self.soft_positives_per_query.append(self.ang_dist)

            #Shallow MLP Training Database
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_dist_positives_per_db = knn.radius_neighbors(self.db_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)

            self.soft_positives_per_db = []

            for i in range(len(self.db_gt_arr)):                 #iterate over all q_gt_array
                self.ang_dist = []
                for j in range(len(self.soft_dist_positives_per_db[i])): #iterate over all positive queries
                    # print(self.q_gt_arr - self.db_gt_arr[self.soft_positives_per_db[i][j]])
                    ang_diff = np.mean(np.abs(self.q_gt_arr_euler[i] - self.db_gt_arr_euler[self.soft_dist_positives_per_db[i][j]]))
                    if ang_diff<self.ang_thresh:
                        self.ang_dist.append(self.soft_dist_positives_per_db[i][j])
                self.soft_positives_per_db.append(self.ang_dist)

        else :
            # Find soft_positives_per_query, which are within val_positive_dist_threshold only
            # self.db_gt_arr = self.db_gt_arr.tolist()
            # self.q_gt_arr = self.q_gt_arr.tolist()
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist_per_query,self.soft_positives_per_query = knn.radius_neighbors(self.q_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)            


            #Shallow MLP Training for database
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_gt_arr)
            self.dist,self.soft_positives_per_db = knn.radius_neighbors(self.db_gt_arr,
                                                                radius=self.dist_thresh,
                                                                return_distance=True)            


        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index]).convert("RGB")
        img = self.transform(img)
        c, h, w = img.shape
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        img = transforms.functional.resize(img, [h, w], antialias=True)

        return img, index
    
    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< #queries: {self.q_num}; #database: {self.db_num} >"

    def get_positives(self):
        if hasattr(self, 'soft_positives_per_query'):
            return self.soft_positives_per_query
        
    def get_positives_database(self):
        return self.soft_positives_per_db
        
if __name__=="__main__":

    vpr_ds = BaiduDataset()
    vg_dl = DataLoader(vpr_ds, 1, pin_memory=True, shuffle=False)
    # # print(vpr_ds.soft_positives_per_query[0])

    # for idx, data in enumerate(vg_dl):
    #     print(data["img_metas"]["org_img_size"])
    #     print(data["img_metas"]["img_size"])

    q_idx = 3
    print(vpr_ds.q_paths[q_idx])
    print(len(vpr_ds.soft_positives_per_query[q_idx]))#,len(vpr_ds.soft_dist_positives_per_query[q_idx]))
    print(len(vpr_ds.dist_per_query[q_idx]))
    num = 0
    for i in range(len(vpr_ds.soft_positives_per_query[q_idx])):
        print(vpr_ds.db_paths[vpr_ds.soft_positives_per_query[q_idx][i]], vpr_ds.dist_per_query[q_idx][i])
        num += 1

    print(num)

    print(vpr_ds.q_gt_arr.shape)
    print(vpr_ds.db_gt_arr.shape)
    print(vpr_ds.soft_positives_per_query.shape) # 2292
    print(vpr_ds.dist_per_query.shape) #
    #     print(vpr_ds.dist[q_idx][i])

    # db_descs, qu_descs, pos_pq = build_cache(largs, vpr_dl, model)
