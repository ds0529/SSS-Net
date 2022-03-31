import numpy as np
import glob, os, sys, shutil, argparse
from os.path import join

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import read_ply
# from helper_tool import Plot


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)

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
     [234, 234, 230]], dtype=np.uint8)

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
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

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


if __name__ == '__main__':
    method_name = 'S3DIS/ablation_method/randlanet_20_rgRGB'
    Log_file = open(join('test', method_name, 'test_Area5.txt'), 'a')
    all_val_preds_dir = join('test', method_name, 'val_preds_Area5')
    os.makedirs(all_val_preds_dir) if not os.path.exists(all_val_preds_dir) else None
    logs = np.sort([os.path.join('test', method_name, f) for f in os.listdir(join('test', method_name)) if
                    f.startswith('Log')])
    for log in logs:
        if log[-6:-1] != 'Area_':
            continue
        log_out(log, Log_file)
        for file in os.listdir(join(log, 'val_preds')):
            shutil.copy(join(log, 'val_preds', file), all_val_preds_dir)

    base_dir = all_val_preds_dir
    saving_path = join(os.path.dirname(all_val_preds_dir), 'visual_ply_Area5')
    os.makedirs(saving_path) if not os.path.exists(saving_path) else None

    original_data_dir = '/home/newdisk/dengshuang/S3DIS/original_ply'
    data_path = glob.glob(os.path.join(base_dir, '*.ply'))
    data_path = np.sort(data_path)
    print(data_path)

    test_total_correct = 0
    test_total_seen = 0
    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]
    visualization = True

    for file_name in data_path:
        pred_data = read_ply(file_name)
        pred = pred_data['pred']
        original_data = read_ply(os.path.join(original_data_dir, file_name.split('/')[-1][:-4] + '.ply'))
        labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T

        ##################
        # Visualize data #
        ##################
        if visualization:
            colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            gt_color = color_table[labels.astype(np.int32), :]
            pred_color = color_table[pred.astype(np.int32), :]
            write_ply(join(saving_path, file_name.split('/')[-1][:-4] + '_origin.ply'), [points, colors],
                      ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
            write_ply(join(saving_path, file_name.split('/')[-1][:-4] + '_gt.ply'), [points, gt_color],
                      ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
            write_ply(join(saving_path, file_name.split('/')[-1][:-4] + '_pred.ply'), [points, pred_color],
                      ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
            # colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            # xyzrgb = np.concatenate([points, colors], axis=-1)
            # Plot.draw_pc(xyzrgb)  # visualize raw point clouds
            # Plot.draw_pc_sem_ins(points, labels)  # visualize ground-truth
            # Plot.draw_pc_sem_ins(points, pred)  # visualize prediction


        correct = np.sum(pred == labels)
        log_out(str(file_name.split('/')[-1][:-4]) + '_acc:' + str(correct / float(len(labels))), Log_file)
        test_total_correct += correct
        test_total_seen += len(labels)

        for j in range(len(labels)):
            gt_l = int(labels[j])
            pred_l = int(pred[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    iou_list = []
    for n in range(13):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou_list.append(iou)
    mean_iou = sum(iou_list) / 13.0
    log_out('eval accuracy: {}'.format(test_total_correct / float(test_total_seen)), Log_file)
    log_out('mean IOU:{}'.format(mean_iou), Log_file)
    log_out(str(iou_list), Log_file)

    acc_list = []
    for n in range(13):
        acc = true_positive_classes[n] / float(gt_classes[n])
        acc_list.append(acc)
    mean_acc = sum(acc_list) / 13.0
    log_out('mAcc value is :{}'.format(mean_acc), Log_file)
