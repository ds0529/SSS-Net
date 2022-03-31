from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time, math, os, pdb, sys
import colorsys, random


color_table = np.array(
    [[231, 231, 91],
     [103, 157, 200],
     [177, 116, 76],
     [88, 164, 149],
     [236, 150, 130],
     [80, 176, 70],
     [108, 136, 66],
     [78, 78, 75],
     [41, 44, 104],
     [217, 49, 50],
     [87, 40, 96],
     [85, 109, 115],
     [234, 234, 230],
     [0, 0, 0]], dtype=np.uint8)


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.

    Examples
    --------

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def stats_graph(graph, log_file):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    log_out('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters), log_file)


def restore_into_scope(model_path, scope_name, sess):
    global_vars = tf.global_variables()
    tensors_to_load = [v for v in global_vars if v.name.startswith(scope_name + '/')]

    load_dict = {}
    for j in range(0, np.size(tensors_to_load)):
        tensor_name = tensors_to_load[j].name
        tensor_name = tensor_name[0:-2] # remove ':0'
        tensor_name = tensor_name.replace(scope_name + '/', 'layers/') #remove scope
        load_dict.update({tensor_name: tensors_to_load[j]})
    loader = tf.train.Saver(var_list=load_dict)
    loader.restore(sess, model_path)
    print("Model restored from: {0} into scope: {1}.".format(model_path, scope_name))


class Network:
    def __init__(self, dataset, config):
        self.sp_switch = 2
        self.edge_switch = 1
        self.sp_agg_switch = 0
        self.sp_loss_switch = 1
        if self.sp_switch == 0:
            self.method_name = 'randlanet_' + str(config.fs_ratio) + '_rg'
        elif self.sp_switch == 1:
            self.method_name = 'randlanet_' + str(config.fs_ratio) + '_rgRGB'
        else:
            self.method_name = 'randlanet_' + str(config.fs_ratio) + '_rg_combine'
        if self.edge_switch:
            self.method_name = self.method_name + '_edge'
        if self.sp_agg_switch:
            self.method_name = self.method_name + '_spagg'
        if self.sp_loss_switch:
            self.method_name = self.method_name + '_sl'

        self.flat_inputs = dataset.flat_inputs  # 包括每层点数,每次调用都直接得到下一个Batch
        self.flat_inputs = list(self.flat_inputs)
        print(self.flat_inputs)
        print(len(self.flat_inputs))
        # pdb.set_trace()

        self.config = config
        # Path of the result folder
        if self.config.saving:  # 存网络
            if self.config.saving_path is None:
                self.saving_path = join('results', self.method_name, time.strftime('Log_%Y-%m-%d_', time.gmtime())
                                        + dataset.name + '_' + str(dataset.val_split))
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.get_input_placeholder()
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = self.placeholder_list[:num_layers]  # 每层的点坐标
            self.inputs['neigh_idx'] = self.placeholder_list[num_layers: 2 * num_layers]  # knn的idx
            self.inputs['sub_idx'] = self.placeholder_list[2 * num_layers:3 * num_layers]  # pooling的idx
            self.inputs['interp_idx'] = self.placeholder_list[3 * num_layers:4 * num_layers]  # upsample的idx
            self.inputs['features'] = self.placeholder_list[4 * num_layers]  # 加入坐标的特征
            self.inputs['labels'] = self.placeholder_list[4 * num_layers + 1]
            self.inputs['rg'] = self.placeholder_list[4 * num_layers + 2]
            self.inputs['rgRGB'] = self.placeholder_list[4 * num_layers + 3]
            self.inputs['rg_combine'] = self.placeholder_list[4 * num_layers + 4]
            self.inputs['input_inds'] = self.placeholder_list[4 * num_layers + 5]  # 点的idx
            self.inputs['cloud_inds'] = self.placeholder_list[4 * num_layers + 6]  # 点云的idx
            self.inputs['fs_switchs'] = self.placeholder_list[4 * num_layers + 7]  # 点云是否全监督

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            if self.sp_agg_switch == 1 or self.sp_loss_switch == 1:
                self.batch_sp_agg_indices_input = tf.placeholder(tf.int32, [None, None, self.config.k_sp])
                self.batch_sp_agg_indices = self.batch_sp_agg_indices_input
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            if self.config.saving:
                self.Log_file = open(self.saving_path + '/log_train_' + dataset.name + '_' + str(dataset.val_split) + '.txt', 'a')

        # 网络结构
        self.batch_size = tf.shape(self.labels)[0]
        self.get_labeled_inputs()
        self.get_unlabeled_inputs()
        with tf.variable_scope('layers'):
            self.logits, self.feat = self.inference(self.inputs, self.is_training, 'layers')
        with tf.variable_scope('pseudo_layers', reuse=False):
            self.pseudo_logits, self.pseudo_feats = self.inference(self.unlabeled_inputs, self.is_training, 'pseudo_layers')

        # sp loss
        self.sp_loss = tf.constant(0, dtype=tf.float32)
        if self.sp_loss_switch:
            agg_feat = self.gather_neighbour(self.feat, self.batch_sp_agg_indices)  # B,N,K,C
            agg_feat_mean = tf.reduce_mean(agg_feat, axis=2, keep_dims=True)  # B,N,1,C
            agg_feat_mean = tf.tile(agg_feat_mean, [1, 1, self.config.k_sp, 1])
            self.sp_loss = tf.reduce_mean(tf.reduce_mean(tf.pow(agg_feat - agg_feat_mean, 2), axis=-1), axis=-1)
            if self.sp_switch == 0:
                masks = self.inputs['rg']
            elif self.sp_switch == 1:
                masks = self.inputs['rg_RGB']
            else:
                masks = self.inputs['rg_combine']
            masks = tf.cast(tf.logical_not(tf.equal(masks, -1)), tf.float32)  # B,N
            # B,N
            self.sp_loss = masks * self.sp_loss
            self.sp_loss = tf.reduce_sum(self.sp_loss) / (tf.reduce_sum(masks)+1e-6)

        # edge loss
        with tf.variable_scope('edge_loss'):
            self.rg_edge, self.rgRGB_edge = self.get_rg_rgRGB_edge(self.inputs)
            self.edge_label = tf.stack([self.rg_edge, self.rgRGB_edge], axis=-1)
            self.edge_label = tf.reshape(self.edge_label, [-1])
            self.edge_label = tf.cast(self.edge_label, dtype=tf.float32)
            self.edge_logits = self.edge_classification_layer(self.feat)
            self.edge_logits = tf.reshape(self.edge_logits, [-1])
            self.edge_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.edge_label, logits=self.edge_logits)
            self.edge_loss = tf.reduce_mean(self.edge_loss)

        # unlabeled loss
        with tf.variable_scope('unsupervised_loss'):
            self.unlabeled_logits = self.logits[self.batch_size // 2:, ...]
            self.unlabeled_labels = tf.cast(tf.argmax(self.pseudo_logits, axis=-1), tf.int32)
            # get unlabeled masks
            self.pseudo_labels_input = tf.placeholder(tf.int32, [None, None])
            self.unlabeled_masks_input = tf.placeholder(tf.int32, [None, None])
            self.pseudo_labels = self.pseudo_labels_input
            self.unlabeled_masks = self.unlabeled_masks_input
            # loss
            self.unlabeled_logits = tf.reshape(self.unlabeled_logits, [-1, config.num_classes])
            self.pseudo_labels = tf.reshape(self.pseudo_labels, [-1])
            self.unlabeled_masks = tf.cast(self.unlabeled_masks, tf.float32)
            self.unlabeled_masks = tf.reshape(self.unlabeled_masks, [-1])

            # # Boolean mask of points that should be ignored
            # unlabeled_ignored_bool = tf.zeros_like(self.pseudo_labels, dtype=tf.bool)
            # for ign_label in self.config.ignored_label_inds:
            #     unlabeled_ignored_bool = tf.logical_or(unlabeled_ignored_bool, tf.equal(self.pseudo_labels, ign_label))
            #
            # # Collect logits and labels that are not ignored
            # unlabeled_valid_idx = tf.squeeze(tf.where(tf.logical_not(unlabeled_ignored_bool)))
            # unlabeled_valid_logits = tf.gather(self.unlabeled_logits, unlabeled_valid_idx, axis=0)
            # unlabeled_valid_masks = tf.gather(self.unlabeled_masks, unlabeled_valid_idx, axis=0)
            # unlabeled_valid_labels_init = tf.gather(self.pseudo_labels, unlabeled_valid_idx, axis=0)  # 只把输出和标签中有效的部分提取出来
            #
            # # Reduce label values in the range of logit shape
            # reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            # inserted_value = tf.zeros((1,), dtype=tf.int32)
            # for ign_label in self.config.ignored_label_inds:
            #     reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            # unlabeled_valid_labels = tf.gather(reducing_list, unlabeled_valid_labels_init)  # 把忽略掉的label值去掉后其他的顺移

            self.unlabeled_supervised_loss = self.get_unlabeled_loss(self.unlabeled_logits, self.pseudo_labels,
                                                                     self.unlabeled_masks, self.class_weights)
            # self.unlabeled_supervised_loss = self.get_unlabeled_loss(unlabeled_valid_logits, unlabeled_valid_labels,
            #                                                          unlabeled_valid_masks, self.class_weights)

        # labeled loss
        with tf.variable_scope('supervised_loss'):
            self.labeled_logits = self.logits[:self.batch_size//2, ...]
            self.labeled_labels = self.labels[:self.batch_size//2, ...]
            self.labeled_logits = tf.reshape(self.labeled_logits, [-1, config.num_classes])
            self.labeled_labels = tf.reshape(self.labeled_labels, [-1])

            # Boolean mask of points that should be ignored
            labeled_ignored_bool = tf.zeros_like(self.labeled_labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                labeled_ignored_bool = tf.logical_or(labeled_ignored_bool, tf.equal(self.labeled_labels, ign_label))

            # Collect logits and labels that are not ignored
            labeled_valid_idx = tf.squeeze(tf.where(tf.logical_not(labeled_ignored_bool)))
            labeled_valid_logits = tf.gather(self.labeled_logits, labeled_valid_idx, axis=0)
            labeled_valid_labels_init = tf.gather(self.labeled_labels, labeled_valid_idx, axis=0)  # 只把输出和标签中有效的部分提取出来

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            labeled_valid_labels = tf.gather(reducing_list, labeled_valid_labels_init)  # 把忽略掉的label值去掉后其他的顺移

            self.labeled_supervised_loss = self.get_loss(labeled_valid_logits, labeled_valid_labels, self.class_weights)

        self.loss = self.labeled_supervised_loss + self.unlabeled_supervised_loss
        if self.edge_switch == 1:
            self.loss = self.loss + self.edge_loss
        if self.sp_loss_switch:
            self.loss = self.loss + self.sp_loss
        self.logits = tf.reshape(self.logits, [-1, config.num_classes])
        self.labels = tf.reshape(self.labels, [-1])

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='layers')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.vars_list)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.save_model_ops = self.save_model('layers', 'pseudo_layers')
            # pdb.set_trace()

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(labeled_valid_logits, labeled_valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=5)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        #####################################################################
        # pretrained model #
        #####################################################################
        if config.pretrained_model:
            chosen_snapshot = -1
            if self.sp_agg_switch:
                logs = np.sort([os.path.join('results', dataset.name, 'randlanet_' + str(self.config.fs_ratio) + '_2', f) for f in
                                os.listdir(join('results', 'randlanet_' + str(self.config.fs_ratio) + '_2')) if
                                f.startswith('Log')])
            else:
                logs = np.sort([os.path.join('results', dataset.name, 'randlanet_' + str(self.config.fs_ratio), f) for f in
                                os.listdir(join('results', 'randlanet_' + str(self.config.fs_ratio))) if
                                f.startswith('Log')])
            # print(logs)
            # for log in logs:
            #     # print(log)
            #     if log[-1] == str(dataset.test_area) and log[-6:-1] == 'Area_':
            #         chosen_folder = log
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap1 = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
            print(chosen_snap1)

            chosen_snapshot = -1
            if self.sp_agg_switch:
                logs = np.sort([os.path.join('results', 'randlanet_' + str(self.config.fs_ratio) + '_2', f) for f in
                                os.listdir(join('results', 'randlanet_' + str(self.config.fs_ratio) + '_2')) if
                                f.startswith('Log')])
            else:
                logs = np.sort([os.path.join('results', 'randlanet_' + str(self.config.fs_ratio), f) for f in
                                os.listdir(join('results', 'randlanet_' + str(self.config.fs_ratio))) if
                                f.startswith('Log')])
            # print(logs)
            # for log in logs:
            #     # print(log)
            #     if log[-1] == str(dataset.test_area) and log[-6:-1] == 'Area_':
            #         chosen_folder = log
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap2 = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
            print(chosen_snap2)
            # pdb.set_trace()

            if chosen_snap1 is not None:
                restore_into_scope(chosen_snap1, 'layers', self.sess)
                restore_into_scope(chosen_snap2, 'pseudo_layers', self.sess)
            # pdb.set_trace()

    def save_model(self, student_scope, teacher_scope):
        teacher_vars = tf.trainable_variables(scope=teacher_scope)
        student_vars = tf.trainable_variables(scope=student_scope)
        save_ops = []
        for var, teacher_var in zip(student_vars, teacher_vars):
            update_op = teacher_var.assign(tf.identity(var))
            save_ops.append(update_op)
        return save_ops

    # inputs
    def get_input_placeholder(self):
        num_layers = self.config.num_layers
        self.placeholder_list = []
        for i in range(num_layers):
            self.placeholder_list.append(tf.placeholder(tf.float32, [None, None, 3]))  # xyz
        for i in range(num_layers):
            self.placeholder_list.append(tf.placeholder(tf.int32))  # neigh_idx
        for i in range(num_layers):
            self.placeholder_list.append(tf.placeholder(tf.int32))  # sub_idx
        for i in range(num_layers):
            self.placeholder_list.append(tf.placeholder(tf.int32))  # interp_idx
        self.placeholder_list.append(tf.placeholder(tf.float32, [None, None, 6]))  # feature
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # labels
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # rg
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # rgRGB
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # rg_combine
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # input_inds
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # cloud_inds
        self.placeholder_list.append(tf.placeholder(tf.int32, [None, None]))  # fs_switchs
        return 1

    def get_feed_dict(self, numpy_list, isTraining=True):
        feed_dictionory = {}
        list_len = len(numpy_list)
        for i in range(list_len):
            feed_dictionory[self.placeholder_list[i]] = numpy_list[i]
        feed_dictionory[self.is_training] = isTraining
        return feed_dictionory

    def get_labeled_inputs(self):
        self.labeled_inputs = dict()
        num_layers = self.config.num_layers
        self.labeled_inputs['xyz'] = []  # 每层的点坐标
        self.labeled_inputs['neigh_idx'] = []  # knn的idx
        self.labeled_inputs['sub_idx'] = []  # pooling的idx
        self.labeled_inputs['interp_idx'] = []  # upsample的idx
        for i in range(num_layers):
            self.labeled_inputs['xyz'].append(self.inputs['xyz'][i][:self.batch_size//2, ...])  # 每层的点坐标
            self.labeled_inputs['neigh_idx'].append(self.inputs['neigh_idx'][i][:self.batch_size//2, ...])  # knn的idx
            self.labeled_inputs['sub_idx'].append(self.inputs['sub_idx'][i][:self.batch_size//2, ...])  # pooling的idx
            self.labeled_inputs['interp_idx'].append(self.inputs['interp_idx'][i][:self.batch_size//2, ...])  # upsample的idx
        self.labeled_inputs['features'] = self.inputs['features'][:self.batch_size//2, ...]  # 加入坐标的特征
        self.labeled_inputs['labels'] = self.inputs['labels'][:self.batch_size//2, ...]
        self.labeled_inputs['rg'] = self.inputs['rg'][:self.batch_size // 2, ...]
        self.labeled_inputs['rgRGB'] = self.inputs['rgRGB'][:self.batch_size // 2, ...]
        self.labeled_inputs['rg_combine'] = self.inputs['rg_combine'][:self.batch_size // 2, ...]
        self.labeled_inputs['input_inds'] = self.inputs['input_inds'][:self.batch_size//2, ...]  # 点的idx
        self.labeled_inputs['cloud_inds'] = self.inputs['cloud_inds'][:self.batch_size//2, ...]  # 点云的idx
        self.labeled_inputs['fs_switchs'] = self.inputs['fs_switchs'][:self.batch_size//2, ...]  # 点云是否全监督
        return 1

    def get_unlabeled_inputs(self):
        self.unlabeled_inputs = dict()
        num_layers = self.config.num_layers
        self.unlabeled_inputs['xyz'] = []  # 每层的点坐标
        self.unlabeled_inputs['neigh_idx'] = []  # knn的idx
        self.unlabeled_inputs['sub_idx'] = []  # pooling的idx
        self.unlabeled_inputs['interp_idx'] = []  # upsample的idx
        for i in range(num_layers):
            self.unlabeled_inputs['xyz'].append(self.inputs['xyz'][i][self.batch_size//2:, ...])  # 每层的点坐标
            self.unlabeled_inputs['neigh_idx'].append(self.inputs['neigh_idx'][i][self.batch_size//2:, ...])  # knn的idx
            self.unlabeled_inputs['sub_idx'].append(self.inputs['sub_idx'][i][self.batch_size//2:, ...])  # pooling的idx
            self.unlabeled_inputs['interp_idx'].append(self.inputs['interp_idx'][i][self.batch_size//2:, ...])  # upsample的idx
        self.unlabeled_inputs['features'] = self.inputs['features'][self.batch_size//2:, ...]  # 加入坐标的特征
        self.unlabeled_inputs['labels'] = self.inputs['labels'][self.batch_size//2:, ...]
        self.unlabeled_inputs['rg'] = self.inputs['rg'][self.batch_size // 2:, ...]
        self.unlabeled_inputs['rgRGB'] = self.inputs['rgRGB'][self.batch_size // 2:, ...]
        self.unlabeled_inputs['rg_combine'] = self.inputs['rg_combine'][self.batch_size // 2:, ...]
        self.unlabeled_inputs['input_inds'] = self.inputs['input_inds'][self.batch_size//2:, ...]  # 点的idx
        self.unlabeled_inputs['cloud_inds'] = self.inputs['cloud_inds'][self.batch_size//2:, ...]  # 点云的idx
        self.unlabeled_inputs['fs_switchs'] = self.inputs['fs_switchs'][self.batch_size//2:, ...]  # 点云是否全监督
        return 1

    # get pseudo labels and spagg
    def get_unlabeled_mask(self, pseudo_labels, unlabeled_sp_labels):
        unlabeled_mask = np.zeros_like(pseudo_labels, dtype=np.int32)
        for bi in range(pseudo_labels.shape[0]):  # batch index
            # find all sp
            uni_sp = np.unique(unlabeled_sp_labels[bi])
            sp_num = uni_sp.shape[0]
            for si in range(sp_num):
                if uni_sp[si] == -1:
                    continue
                # find sp
                # sp_mask = np.equal(sp_labels[bi], uni_sp[si])
                sp_indics = np.where(unlabeled_sp_labels[bi] == uni_sp[si])[0]
                points_in_sp = sp_indics.shape[0]
                labels_in_sp = pseudo_labels[bi][sp_indics]
                # find label in sp
                labels_count = np.bincount(labels_in_sp)
                label = np.argmax(labels_count)
                label_points = labels_count[label]
                ratio = float(label_points) / float(points_in_sp)
                # mask
                if ratio > 0.8:
                    unlabeled_mask[bi][sp_indics] = 1
                    pseudo_labels[bi][sp_indics] = label
        return pseudo_labels, unlabeled_mask

    def get_spagg(self, sp_labels):
        batch_sp_agg_indices = np.zeros([sp_labels.shape[0], sp_labels.shape[1], self.config.k_sp],
                                        dtype=np.int32)
        for bi in range(sp_labels.shape[0]):  # batch index
            # find all sp
            uni_sp = np.unique(sp_labels[bi])
            sp_num = uni_sp.shape[0]
            for si in range(sp_num):
                if uni_sp[si] == -1:
                    continue
                # find sp
                # sp_mask = np.equal(sp_labels[bi], uni_sp[si])
                sp_indics = np.where(sp_labels[bi] == uni_sp[si])[0]
                # get spagg
                if sp_indics.shape[0] < self.config.k_sp:
                    num_in = sp_indics.shape[0]
                    # dup = np.random.choice(num_in, self.config.k_sp - num_in)
                    # sp_indics_dup = sp_indics[dup]
                    # sp_agg_indices = np.concatenate([sp_indics, sp_indics_dup], 0)
                    sp_agg_indices = np.zeros([sp_indics.shape[0], self.config.k_sp], dtype=np.int32)
                    sp_agg_indices[:, :num_in] = sp_indics
                    dup = np.random.choice(num_in, (self.config.k_sp - num_in) * sp_indics.shape[0])
                    sp_indics_dup = sp_indics[dup]
                    sp_indics_dup = np.reshape(sp_indics_dup, [sp_indics.shape[0], self.config.k_sp - num_in])
                    sp_agg_indices[:, num_in:] = sp_indics_dup
                elif sp_indics.shape[0] > self.config.k_sp:
                    num_in = sp_indics.shape[0]
                    # dup = np.random.choice(num_in, self.config.k_sp)
                    # sp_agg_indices = sp_indics[dup]
                    dup = np.random.choice(num_in, self.config.k_sp * sp_indics.shape[0])
                    sp_indics_dup = sp_indics[dup]
                    sp_indics_dup = np.reshape(sp_indics_dup, [sp_indics.shape[0], self.config.k_sp])
                    sp_agg_indices = sp_indics_dup
                else:
                    sp_agg_indices = sp_indics
                batch_sp_agg_indices[bi, sp_indics, :] = sp_agg_indices
        return batch_sp_agg_indices

    # get edges
    def get_rg_rgRGB_edge(self, inputs):
        neighbor_num = 6
        rg_label = inputs['rg']  # BxN
        rgRGB_label = inputs['rgRGB']  # BxN

        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(rg_label)[0]
        num_points = tf.shape(rg_label)[1]
        index_input = tf.reshape(inputs['neigh_idx'][0], shape=[batch_size, -1])
        rg_neigh = tf.batch_gather(rg_label, index_input)
        rg_neigh = tf.reshape(rg_neigh, [batch_size, num_points, tf.shape(inputs['neigh_idx'][0])[-1]])
        rgRGB_neigh = tf.batch_gather(rgRGB_label, index_input)
        rgRGB_neigh = tf.reshape(rgRGB_neigh, [batch_size, num_points, tf.shape(inputs['neigh_idx'][0])[-1]])
        rg_neigh = rg_neigh[:, :, :neighbor_num]
        rgRGB_neigh = rgRGB_neigh[:, :, :neighbor_num]

        # get edge
        rg_label_tile = tf.tile(tf.expand_dims(rg_label, axis=-1), [1, 1, neighbor_num])
        rg_edge = tf.reduce_sum(tf.abs(rg_label_tile - rg_neigh), axis=-1)
        # rg_edge = tf.logical_not(tf.equal(rg_edge, 0))
        # rg_edge_2 = tf.equal(rg_label, -1)
        # rg_edge = tf.cast(tf.logical_or(rg_edge, rg_edge_2), tf.int32)
        rg_edge = tf.equal(rg_edge, 0)
        rg_edge_2 = tf.logical_not(tf.equal(rg_label, -1))
        rg_edge = tf.cast(tf.logical_and(rg_edge, rg_edge_2), tf.int32)

        rgRGB_label_tile = tf.tile(tf.expand_dims(rgRGB_label, axis=-1), [1, 1, neighbor_num])
        rgRGB_edge = tf.reduce_sum(tf.abs(rgRGB_label_tile - rgRGB_neigh), axis=-1)
        # rgRGB_edge = tf.logical_not(tf.equal(rgRGB_edge, 0))
        # rgRGB_edge_2 = tf.equal(rgRGB_label, -1)
        # rgRGB_edge = tf.cast(tf.logical_or(rgRGB_edge, rgRGB_edge_2), tf.int32)
        rgRGB_edge = tf.equal(rgRGB_edge, 0)
        rgRGB_edge_2 = tf.logical_not(tf.equal(rgRGB_label, -1))
        rgRGB_edge = tf.cast(tf.logical_and(rgRGB_edge, rgRGB_edge_2), tf.int32)
        return rg_edge, rgRGB_edge

    # model
    def inference(self, inputs, is_training, scope_name):

        d_out = self.config.d_out  # 每层特征维度
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')  # 全连接
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)  # B,N,1,8

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)  # 残差块,这里没有下采样的步骤
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])  # 下采样, max pooling
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)  # 不改变维度的MLP

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])  # 内插特征
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)  # 串接并MLP
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)

        if self.sp_agg_switch:
            in_feat = tf.squeeze(f_layer_fc2, axis=2)
            d = in_feat.get_shape()[-1].value

            if scope_name == 'layers':
                agg_feat = self.gather_neighbour(in_feat, self.batch_sp_agg_indices)  # B,N,K,C
            else:
                agg_feat = self.gather_neighbour(in_feat,
                                                 self.batch_sp_agg_indices[(self.batch_size // 2):, ...])  # B,N,K,C
            agg_feat = tf.reduce_mean(agg_feat, axis=2, keep_dims=True)  # B,N,1,C

            if self.sp_switch == 0:
                masks = inputs['rg']
            elif self.sp_switch == 1:
                masks = inputs['rg_RGB']
            else:
                masks = inputs['rg_combine']
            masks = tf.logical_not(tf.equal(masks, -1))  # B,N
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.tile(masks, [1, 1, 1, d])  # B,N,1,C
            # agg_feat = agg_feat * masks
            in_feat = tf.expand_dims(in_feat, axis=2)
            agg_feat = tf.where(masks, agg_feat, in_feat)
            out_feat = (in_feat + agg_feat) / 2
            f_layer_fc2 = out_feat

        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        feature = tf.squeeze(f_layer_fc2, [2])
        return f_out, feature

    def sp_aggregate_layer(self, inputs, in_feat, sp_agg_indices, name, is_training):
        d = in_feat.get_shape()[-1].value

        # agg_feat = self.gather_neighbour(in_feat, sp_agg_indices)  # B,N,K,C
        # agg_feat = tf.reduce_mean(agg_feat, axis=2, keep_dims=True)
        #
        # if self.sp_switch == 0:
        #     masks = inputs['rg']
        # elif self.sp_switch == 1:
        #     masks = inputs['rg_RGB']
        # else:
        #     masks = inputs['rg_combine']
        # masks = tf.logical_not(tf.equal(masks, -1))  # B,N
        # masks = tf.expand_dims(masks, axis=-1)
        # masks = tf.expand_dims(masks, axis=-1)
        # masks = tf.tile(masks, [1, 1, 1, d])  # B,N,1,d
        # # agg_feat = agg_feat * masks
        # in_feat = tf.expand_dims(in_feat, axis=2)
        # agg_feat = tf.where(masks, agg_feat, in_feat)
        #
        # # out_feat = (in_feat + agg_feat) / 2
        # out_feat = tf.concat([in_feat, agg_feat], axis=-1)
        # out_feat = helper_tf_util.conv2d(out_feat, d, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        # out_feat_drop = helper_tf_util.dropout(out_feat, keep_prob=0.5, is_training=is_training, scope=name + 'dp1')
        # out_logits = helper_tf_util.conv2d(out_feat_drop, self.config.num_classes, [1, 1], name + 'fc', [1, 1], 'VALID', False,
        #                                     is_training, activation_fn=None)
        # out_logits = tf.squeeze(out_logits, [2])
        # out_feat = tf.squeeze(out_feat, [2])

        agg_feat = self.gather_neighbour(in_feat, sp_agg_indices)  # B,N,K,C
        agg_w = helper_tf_util.conv2d(agg_feat, 1, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)  # B,N,K,1
        agg_w = tf.nn.softmax(agg_w, axis=2)  # 变成概率
        agg_feat = helper_tf_util.conv2d(agg_feat, d, [1, 1], name + 'mlp2', [1, 1], 'VALID', True,
                                         is_training)  # B,N,K,1
        agg_feat = tf.reduce_sum(tf.tile(agg_w, [1, 1, 1, d]) * agg_feat, axis=2, keep_dims=True)  # B,N,1,C

        if self.sp_switch == 0:
            masks = inputs['rg']
        elif self.sp_switch == 1:
            masks = inputs['rg_RGB']
        else:
            masks = inputs['rg_combine']
        masks = tf.logical_not(tf.equal(masks, -1))  # B,N
        masks = tf.expand_dims(masks, axis=-1)
        masks = tf.expand_dims(masks, axis=-1)
        masks = tf.tile(masks, [1, 1, 1, d])
        # agg_feat = agg_feat * masks
        in_feat = tf.expand_dims(in_feat, axis=2)
        agg_feat = tf.where(masks, agg_feat, in_feat)

        # out_feat = (in_feat + agg_feat) / 2
        out_feat = tf.concat([in_feat, agg_feat], axis=-1)
        out_feat = helper_tf_util.conv2d(out_feat, d, [1, 1], name + 'mlp3', [1, 1], 'VALID', True, is_training)
        out_feat_drop = helper_tf_util.dropout(out_feat, keep_prob=0.5, is_training=is_training, scope=name + 'dp1')
        out_logits = helper_tf_util.conv2d(out_feat_drop, self.config.num_classes, [1, 1], name + 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        out_logits = tf.squeeze(out_logits, [2])
        out_feat = tf.squeeze(out_feat, [2])

        return out_logits, out_feat

    def edge_classification_layer(self, x, name="edge_classification", reuse=False, isTrainable=True):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            # net = tf.layers.dense(inputs=x, units=64,
            #                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
            #                       activation=tf.nn.leaky_relu, name='fc1', trainable=isTrainable, reuse=reuse)
            # net = tf.layers.dense(inputs=net, units=32,
            #                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
            #                       activation=tf.nn.leaky_relu, name='fc2', trainable=isTrainable, reuse=reuse)
            net = tf.layers.dense(inputs=x, units=2,
                                        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                        activation=None, name='fc3', trainable=isTrainable, reuse=reuse)
        return net

    # train
    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                # get inputs
                flat_inputs = self.sess.run(self.flat_inputs)
                # get sp inputs
                sp_labels_rg, sp_labels_rgRGB, sp_labels_rg_combine, \
                rg_edge, rgRGB_edge = self.sess.run(
                    [self.inputs['rg'], self.inputs['rgRGB'], self.inputs['rg_combine'],
                     self.rg_edge, self.rgRGB_edge],
                    self.get_feed_dict(flat_inputs, False))
                # get spagg
                if self.sp_agg_switch == 1 or self.sp_loss_switch == 1:
                    if self.sp_switch == 0:
                        batch_sp_agg_indices = self.get_spagg(sp_labels_rg)
                    elif self.sp_switch == 1:
                        batch_sp_agg_indices = self.get_spagg(sp_labels_rgRGB)
                    else:
                        batch_sp_agg_indices = self.get_spagg(sp_labels_rg_combine)
                    feed_dictionary = self.get_feed_dict(flat_inputs, False)
                    feed_dictionary[self.batch_sp_agg_indices_input] = batch_sp_agg_indices
                    unlabeled_labels = self.sess.run(self.unlabeled_labels, feed_dictionary)
                else:
                    unlabeled_labels = self.sess.run(self.unlabeled_labels, self.get_feed_dict(flat_inputs, False))

                # get pseudo_labels
                if self.sp_switch == 0:
                    pseudo_labels, unlabeled_masks = self.get_unlabeled_mask(unlabeled_labels, sp_labels_rg[
                                                                                               self.config.batch_size // 2:, ...])
                elif self.sp_switch == 1:
                    pseudo_labels, unlabeled_masks = self.get_unlabeled_mask(unlabeled_labels, sp_labels_rgRGB[
                                                                                               self.config.batch_size // 2:, ...])
                else:
                    pseudo_labels, unlabeled_masks = self.get_unlabeled_mask(unlabeled_labels,
                                                                             sp_labels_rg_combine[
                                                                             self.config.batch_size // 2:, ...])

                # # display
                # # pseudo labels
                # half_batch = self.config.batch_size // 2
                # pseudo_labels_dis = pseudo_labels.copy()
                # pseudo_labels_dis = np.where(unlabeled_masks > 0, pseudo_labels_dis, -1)
                # pseudo_labels_color = color_table[pseudo_labels_dis[0].astype(np.int32), :]
                # write_ply(join('figs', 'labels_pseudo_' + str(self.training_step) + '.ply'),
                #         [flat_inputs[0][half_batch, :, :], pseudo_labels_color],
                #         ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # labels
                # labels = flat_inputs[21][half_batch, :]
                # labels_color = color_table[labels.astype(np.int32), :]
                # write_ply(join('figs', 'labels_gt_' + str(self.training_step) + '.ply'),
                #         [flat_inputs[0][half_batch, :, :], labels_color],
                #         ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # predicts
                # predicts = unlabeled_labels
                # predicts_color = color_table[predicts[0].astype(np.int32), :]
                # write_ply(join('figs', 'labels_predict_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], predicts_color],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # input
                # input_color = flat_inputs[20][half_batch, :, 3:] * 255
                # input_color = input_color.astype(np.uint8)
                # write_ply(join('figs', 'input_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], input_color],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # rg
                # rg_labels = sp_labels_rg[half_batch, :]
                # # rg_labels = np.where(rg_edge[half_batch, :] > 0, rg_labels, -1)
                # ins_colors = random_colors(len(np.unique(rg_labels)) + 1, seed=2)
                # colors = np.zeros((rg_labels.shape[0], 3), np.uint8)
                # uni_index = np.unique(rg_labels)
                # for id, semins in enumerate(uni_index):
                #     valid_ind = np.argwhere(rg_labels == semins)[:, 0]
                #     if semins <= -1:
                #         tp = [0, 0, 0]
                #     else:
                #         tp = np.array(ins_colors[id], dtype=np.float32)
                #         tp[0] = 255 * tp[0]
                #         tp[1] = 255 * tp[1]
                #         tp[2] = 255 * tp[2]
                #     colors[valid_ind] = tp
                # write_ply(join('figs', 'rg_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], colors],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # rgRGB
                # rgRGB_labels = sp_labels_rgRGB[half_batch, :]
                # # rgRGB_labels = np.where(rgRGB_edge[half_batch, :] > 0, rgRGB_labels, -1)
                # ins_colors = random_colors(len(np.unique(rgRGB_labels)) + 1, seed=2)
                # colors = np.zeros((rgRGB_labels.shape[0], 3), np.uint8)
                # uni_index = np.unique(rgRGB_labels)
                # for id, semins in enumerate(uni_index):
                #     valid_ind = np.argwhere(rgRGB_labels == semins)[:, 0]
                #     if semins <= -1:
                #         tp = [0, 0, 0]
                #     else:
                #         tp = np.array(ins_colors[id], dtype=np.float32)
                #         tp[0] = 255 * tp[0]
                #         tp[1] = 255 * tp[1]
                #         tp[2] = 255 * tp[2]
                #     colors[valid_ind] = tp
                # write_ply(join('figs', 'rgRGB_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], colors],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # rg_combine
                # rg_combine_labels = sp_labels_rg_combine[half_batch, :]
                # # rg_combine_labels = np.where(rg_edge[half_batch, :] > 0, rg_combine_labels, -1)
                # # rg_combine_labels = np.where(rgRGB_edge[half_batch, :] > 0, rg_combine_labels, -1)
                # ins_colors = random_colors(len(np.unique(rg_combine_labels)) + 1, seed=2)
                # colors = np.zeros((rg_combine_labels.shape[0], 3), np.uint8)
                # uni_index = np.unique(rg_combine_labels)
                # for id, semins in enumerate(uni_index):
                #     valid_ind = np.argwhere(rg_combine_labels == semins)[:, 0]
                #     if semins <= -1:
                #         tp = [0, 0, 0]
                #     else:
                #         tp = np.array(ins_colors[id], dtype=np.float32)
                #         tp[0] = 255 * tp[0]
                #         tp[1] = 255 * tp[1]
                #         tp[2] = 255 * tp[2]
                #     colors[valid_ind] = tp
                # write_ply(join('figs', 'rg_combine_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], colors],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # # edge
                # edge_color_table = np.array(
                #     [[255, 255, 255],
                #      [0, 0, 0]], dtype=np.uint8)
                # rg_edge_dis = rg_edge.copy()
                # rg_edge_color = edge_color_table[rg_edge_dis[half_batch].astype(np.int32), :]
                # rgRGB_edge_dis = rgRGB_edge.copy()
                # rgRGB_edge_color = edge_color_table[rgRGB_edge_dis[half_batch].astype(np.int32), :]
                # write_ply(join('figs', 'rg_edge_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], rg_edge_color],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # write_ply(join('figs', 'rgRGB_edge_' + str(self.training_step) + '.ply'),
                #           [flat_inputs[0][half_batch, :, :], rgRGB_edge_color],
                #           ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
                # pdb.set_trace()

                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.labeled_supervised_loss,
                       self.unlabeled_supervised_loss,
                       self.edge_loss,
                       self.sp_loss,
                       self.logits,
                       self.labels,
                       self.accuracy,
                       self.inputs['fs_switchs']]
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_feed_dictionary = self.get_feed_dict(flat_inputs, True)
                train_feed_dictionary[self.pseudo_labels_input] = pseudo_labels
                train_feed_dictionary[self.unlabeled_masks_input] = unlabeled_masks
                if self.sp_agg_switch == 1 or self.sp_loss_switch == 1:
                    train_feed_dictionary[self.batch_sp_agg_indices_input] = batch_sp_agg_indices
                _, _, summary, l_out, ll_out, ul_out,el_out, sl_out, probs, labels, acc, fs_switchs = \
                    self.sess.run(ops,
                                  train_feed_dictionary,
                                  options=run_options,
                                  run_metadata=run_metadata)
                # self.train_writer.add_run_metadata(run_metadata, 'step%d' % self.training_step)
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} lL_out={:5.3f} uL_out={:5.3f} eL_out={:5.3f} sL_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, ll_out, ul_out, el_out, sl_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # # update pseudo labels
                # if self.training_epoch % 5 == 0:
                #     self.sess.run(self.save_model_ops)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                flat_inputs = self.sess.run(self.flat_inputs)
                # get sp inputs
                sp_labels_rg, sp_labels_rgRGB, sp_labels_rg_combine = self.sess.run(
                    [self.inputs['rg'], self.inputs['rgRGB'], self.inputs['rg_combine']],
                    self.get_feed_dict(flat_inputs, False))
                # get spagg
                if self.sp_agg_switch == 1 or self.sp_loss_switch == 1:
                    if self.sp_switch == 0:
                        batch_sp_agg_indices = self.get_spagg(sp_labels_rg)
                    elif self.sp_switch == 1:
                        batch_sp_agg_indices = self.get_spagg(sp_labels_rgRGB)
                    else:
                        batch_sp_agg_indices = self.get_spagg(sp_labels_rg_combine)

                ops = (self.prob_logits, self.labels, self.accuracy)
                test_feed_dictionary = self.get_feed_dict(flat_inputs, False)
                if self.sp_agg_switch == 1 or self.sp_loss_switch == 1:
                    test_feed_dictionary[self.batch_sp_agg_indices_input] = batch_sp_agg_indices
                stacked_prob, labels, acc = self.sess.run(ops, test_feed_dictionary)
                pred = np.argmax(stacked_prob, -1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def get_unlabeled_loss(self, logits, labels, mask, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        print(pre_cal_weights.shape)
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        print(class_weights.get_shape())
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        print(one_hot_labels.get_shape())
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        print(weights.get_shape())
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        print(unweighted_losses.get_shape())
        weighted_losses = unweighted_losses * weights * mask
        output_loss = tf.reduce_sum(weighted_losses) / tf.reduce_sum(mask)
        # pdb.set_trace()
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value  # 输入维度
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # 点间位置的encode
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)  # 位置encodeing
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)  # B,N,K,C
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # 连接位置encoding和输入特征
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)  # 加权求和

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)  # 第二个点的encoding
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # 串接
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)  # 加权求和
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # B,N,K,3
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz  # p_j-P_i
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))  # ||p_j-P_i||
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)  # B,N,C
        num_neigh = tf.shape(pool_idx)[-1]  # knn
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)  # max pooling
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')  # 权重
        att_scores = tf.nn.softmax(att_activation, axis=1)  # 变成概率
        f_agg = f_reshaped * att_scores  # 点乘
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)  # MLP
        return f_agg
