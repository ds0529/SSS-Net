from os.path import join
# from RandLANet import Network
from tester_ScanNet_sp import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigScanNet as cfg
from helper_tool import DataProcessing as DP
# from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os, importlib, pdb


class ScanNet:
    def __init__(self):
        self.name = 'ScanNet'
        self.path = '/home/newdisk/dengshuang/scannet/pickle_xyzrgb'
        # self.path = '/share/group/datanlpr9/data/scannet/pickle_xyzrgb'
        self.label_to_names = {0: 'unannotated',
                               1: 'wall',
                               2: 'floor',
                               3: 'chair',
                               4: 'table',
                               5: 'desk',
                               6: 'bed',
                               7: 'bookshelf',
                               8: 'sofa',
                               9: 'sink',
                               10: 'bathtub',
                               11: 'toilet',
                               12: 'curtain',
                               13: 'counter',
                               14: 'door',
                               15: 'window',
                               16: 'shower curtain',
                               17: 'refridgerator',
                               18: 'picture',
                               19: 'cabinet',
                               20: 'otherfurniture'}
        self.num_classes = len(self.label_to_names)  # 13
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # class number
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # label:idx
        self.ignored_labels = np.array([0])

        self.val_split = 1
        self.all_train_files = glob.glob(join(self.path, 'original_ply', 'train', '*.ply'))
        self.all_test_files = glob.glob(join(self.path, 'original_ply', 'test', '*.ply'))
        self.all_files = self.all_train_files + self.all_test_files
        self.fs_files = [line.rstrip() for line in open('utils/ScanNet_fs_samples_' + str(cfg.fs_ratio) + '.txt')]
        self.training_fs_index = []

        # Initiate containers
        self.val_proj = []  # ???????????????????????????????????????????????????
        self.val_labels = []
        self.possibility = {}  # ??????????????????
        self.min_possibility = {}  # ?????????????????????
        self.input_trees = {'training_fs': [], 'training_all': [], 'validation': []}  # ????????????????????????????????????
        self.input_colors = {'training_fs': [], 'training_all': [], 'validation': []}
        self.input_labels = {'training_fs': [], 'training_all': [], 'validation': []}
        self.input_names = {'training_fs': [], 'training_all': [], 'validation': []}
        self.input_rg = {'training_fs': [], 'training_all': [], 'validation': []}
        self.input_rgRGB = {'training_fs': [], 'training_all': [], 'validation': []}
        self.input_rg_combine = {'training_fs': [], 'training_all': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_idx = file_path.split('/')[-1][:-4]  # ?????????
            split_name = file_path.split('/')[-2]
            file_path = join(file_path.split('/')[-2], file_path.split('/')[-1])
            if split_name == 'test':
                cloud_split = 'validation'
            elif split_name == 'train' and file_path in self.fs_files:
                cloud_split = 'training_fs'
            else:
                cloud_split = 'training_all'

            # Name of the input files
            kd_tree_file = join(tree_path, split_name, '{:s}_KDTree.pkl'.format(cloud_idx))  # ???
            sub_ply_file = join(tree_path, split_name, '{:s}.ply'.format(cloud_idx))  # ?????????

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # region growing
            self.rg_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size), 'region_growing', split_name)
            self.rgRGB_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size), 'region_growingRGB', split_name)
            self.rg_combine_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size), 'region_growing_combine', split_name)
            rg_file = join(self.rg_path, '{:s}.txt'.format(cloud_idx))  # ?????????
            rgRGB_file = join(self.rgRGB_path, '{:s}.txt'.format(cloud_idx))  # ?????????
            rg_combine_file = join(self.rg_combine_path, '{:s}.txt'.format(cloud_idx))  # ?????????
            # sub_rg = np.loadtxt(rg_file, dtype=np.int32)
            # sub_rgRGB = np.loadtxt(rgRGB_file, dtype=np.int32)
            f = open(rg_file, 'r')
            sub_rg = np.array(f.readlines())
            f.close()
            f = open(rgRGB_file, 'r')
            sub_rgRGB = np.array(f.readlines())
            f.close()
            f = open(rg_combine_file, 'r')
            sub_rg_combine = np.array(f.readlines())
            f.close()
            # sub_rgRGB = sub_rg
            # print(sub_labels.shape)
            # print(sub_rg.shape)
            # print(sub_rgRGB.shape)
            # pdb.set_trace()

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_idx]
            self.input_rg[cloud_split] += [sub_rg]
            self.input_rgRGB[cloud_split] += [sub_rgRGB]
            self.input_rg_combine[cloud_split] += [sub_rg_combine]
            # if cloud_split == 'training_fs':
            #     self.input_trees['training_all'] += [search_tree]
            #     self.input_colors['training_all'] += [sub_colors]
            #     self.input_labels['training_all'] += [sub_labels]
            #     self.input_names['training_all'] += [cloud_name]
            #     self.input_rg['training_all'] += [sub_rg]
            #     self.input_rgRGB['training_all'] += [sub_rgRGB]
            #     self.input_rg_combine['training_all'] += [sub_rg_combine]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        for i in self.training_fs_index:
            print(self.input_names['training'][i])
        # pdb.set_trace()

        print('\nPreparing reprojected indices for testing')
        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_test_files):
            t0 = time.time()
            cloud_idx = file_path.split('/')[-1][:-4]  # ?????????
            split_name = file_path.split('/')[-2]

            # Validation projection and labels
            proj_file = join(tree_path, split_name, '{:s}_proj.pkl'.format(cloud_idx))
            with open(proj_file, 'rb') as f:
                proj_idx, labels = pickle.load(f)
            self.val_proj += [proj_idx]
            self.val_labels += [labels]
            print('{:s} done in {:.1f}s'.format(cloud_idx, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size  # ??????epoch?????????????????????
            # fs
            self.possibility['training_fs'] = []  # ????????????
            self.min_possibility['training_fs'] = []  # ???????????????
            # Random initialize
            for i, tree in enumerate(self.input_colors['training_fs']):  # ?????????????????????
                self.possibility['training_fs'] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                self.min_possibility['training_fs'] += [float(np.min(self.possibility['training_fs'][-1]))]
            # all
            self.possibility['training_all'] = []  # ????????????
            self.min_possibility['training_all'] = []  # ???????????????
            # Random initialize
            for i, tree in enumerate(self.input_colors['training_all']):  # ?????????????????????
                self.possibility['training_all'] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                self.min_possibility['training_all'] += [float(np.min(self.possibility['training_all'][-1]))]
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
            self.possibility[split] = []  # ????????????
            self.min_possibility[split] = []  # ???????????????
            # Random initialize
            for i, tree in enumerate(self.input_colors[split]):  # ?????????????????????
                self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen_training():
            # Generator loop
            for i in range(num_per_epoch):  # ??????epoch?????????????????????
                if i % cfg.batch_size < cfg.batch_size / 2:
                    split_tmp = 'training_fs'
                else:
                    split_tmp = 'training_all'
                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split_tmp]))  # ??????????????????
                fs_switch = 0
                if split_tmp == 'training_fs':
                    fs_switch = 1

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split_tmp][cloud_idx])  # ???????????????

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split_tmp][cloud_idx].data, copy=False)  # ????????????????????????????????????

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)  # ???????????????????????????

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)  # ???????????????????????????

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split_tmp][cloud_idx].query(pick_point, k=len(points))[1][0]  # ??????????????????????????????????????????
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split_tmp][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)  # ??????
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point  # ?????????
                queried_pc_colors = self.input_colors[split_tmp][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split_tmp][cloud_idx][queried_idx]
                queried_pc_rg = self.input_rg[split_tmp][cloud_idx][queried_idx]
                queried_pc_rgRGB = self.input_rgRGB[split_tmp][cloud_idx][queried_idx]
                queried_pc_rg_combine = self.input_rg_combine[split_tmp][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split_tmp][cloud_idx][queried_idx] += delta  # ?????????????????????????????????
                self.min_possibility[split_tmp][cloud_idx] = float(np.min(self.possibility[split_tmp][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels, queried_pc_rg, queried_pc_rgRGB, queried_pc_rg_combine = \
                        DP.data_aug_rg(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_pc_rg, queried_pc_rgRGB, queried_pc_rg_combine, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_pc_rg,
                           queried_pc_rgRGB,
                           queried_pc_rg_combine,
                           queried_idx.astype(np.int32),  # ???????????????idx
                           np.array([cloud_idx], dtype=np.int32),  # ??????????????????idx
                           np.array([fs_switch], dtype=np.int32))  # ??????????????????????????????

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # ??????epoch?????????????????????

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))  # ??????????????????
                fs_switch = 0
                if split == 'training':
                    if cloud_idx in self.training_fs_index:
                        fs_switch = 1

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])  # ???????????????

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)  # ????????????????????????????????????

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)  # ???????????????????????????

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)  # ???????????????????????????

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]  # ??????????????????????????????????????????
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)  # ??????
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point  # ?????????
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]
                queried_pc_rg = self.input_rg[split][cloud_idx][queried_idx]
                queried_pc_rgRGB = self.input_rgRGB[split][cloud_idx][queried_idx]
                queried_pc_rg_combine = self.input_rg_combine[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta  # ?????????????????????????????????
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels, queried_pc_rg, queried_pc_rgRGB, queried_pc_rg_combine = \
                        DP.data_aug_rg(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_pc_rg, queried_pc_rgRGB, queried_pc_rg_combine, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_pc_rg,
                           queried_pc_rgRGB,
                           queried_pc_rg_combine,
                           queried_idx.astype(np.int32),  # ???????????????idx
                           np.array([cloud_idx], dtype=np.int32),  # ??????????????????idx
                           np.array([fs_switch], dtype=np.int32))  # ??????????????????????????????

        if split == 'training':
            gen_func = spatially_regular_gen_training
        else:
            gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None], [None], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # ????????????????????????????????????
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_rg, batch_rgRGB, batch_rg_combine, batch_pc_idx, batch_cloud_idx, batch_fs_switch):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)  # ?????????????????????
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            # attention_neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.attention_k_n], tf.int32)

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]  # ????????????????????????????????????????????????????????????????????????????????????
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]

                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)  # ?????????????????????????????????????????????
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points  # ?????????

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_rg, batch_rgRGB, batch_rg_combine, batch_pc_idx, batch_cloud_idx, batch_fs_switch]
            # input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, attention_neighbour_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)  # ??????????????????
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)  # ???????????????????????????batch???
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)  # ??????????????????batch?????????????????????????????????
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--model', default='model name')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    MODEL = importlib.import_module(FLAGS.model)  # import network module
    Network = MODEL.Network

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = ScanNet()
    dataset.init_input_pipeline()  # ?????????????????????

    if Mode == 'train':
        model = Network(dataset, cfg)  # ??????
        model.train(dataset)  # ??????
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        method_name = model.method_name
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
            # from tensorflow.python import pywrap_tensorflow
            # reader = pywrap_tensorflow.NewCheckpointReader(chosen_snap)
            # var_to_shape_map = reader.get_variable_to_shape_map()
            # for key in var_to_shape_map:
            #     print('tensor_name: ', key)
            # pdb.set_trace()
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', method_name, f) for f in os.listdir(join('results', method_name)) if f.startswith('Log')])
            # print(logs)
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        print(chosen_snap)
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    # else:
    #     ##################
    #     # Visualize data #
    #     ##################
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(dataset.train_init_op)
    #         while True:
    #             flat_inputs = sess.run(dataset.flat_inputs)
    #             pc_xyz = flat_inputs[0]
    #             sub_pc_xyz = flat_inputs[1]
    #             labels = flat_inputs[21]
    #             Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
    #             Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
