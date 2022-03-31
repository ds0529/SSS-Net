import numpy as np
import glob, os, sys, shutil, argparse
from os.path import join
import colorsys, random, pdb

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

# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


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


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file
    """

    with open(filename, 'rb') as plyfile:

        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors


dataset = 'semantic3d'
if dataset == 's3dis':
    rg_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growing\\'
    rgrgb_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growingRGB\\'
    ply_path = 'F:\\datasets\\S3DIS\\input_0.040\\'
    save_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growing_combine\\'
    save_ply_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growing_combine_ply\\'
elif dataset == 'scannet_train':
    rg_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing\\train\\'
    rgrgb_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growingRGB\\train\\'
    ply_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\train'
    save_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing_combine\\train\\'
    save_ply_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing_combine_ply\\train\\'
elif dataset == 'scannet_test':
    rg_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing\\test\\'
    rgrgb_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growingRGB\\test\\'
    ply_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\test'
    save_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing_combine\\test\\'
    save_ply_path = 'F:\\datasets\\ScanNet\\scans\\input_0.020\\region_growing_combine_ply\\test\\'
elif dataset == 'semantic3d':
    rg_path = 'F:\\datasets\\Semantic3D\\input_0.060\\region_growing\\'
    rgrgb_path = 'F:\\datasets\\Semantic3D\\input_0.060\\region_growingRGB\\'
    ply_path = 'F:\\datasets\\Semantic3D\\input_0.060\\'
    save_path = 'F:\\datasets\\Semantic3D\\input_0.060\\region_growing_combine\\'
    save_ply_path = 'F:\\datasets\\Semantic3D\\input_0.060\\region_growing_combine_ply\\'
else:
    rg_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growing\\'
    rgrgb_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growingRGB\\'
    ply_path = 'F:\\datasets\\S3DIS\\input_0.040\\'
    save_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growing_combine\\'
    save_ply_path = 'F:\\datasets\\S3DIS\\input_0.040\\region_growing_combine_ply\\'

rg_files = os.listdir(rg_path)
rgrgb_files = os.listdir(rgrgb_path)
for rg_file in rg_files:
    # input
    rg_input_file = join(rg_path, rg_file)
    rg_label = np.loadtxt(rg_input_file, dtype=np.int32, delimiter="\n")
    rgrgb_input_file = join(rgrgb_path, rg_file)
    rgrgb_label = np.loadtxt(rgrgb_input_file, dtype=np.int32, delimiter="\n")
    print(rg_label.shape)
    print(rgrgb_label.shape)
    # pdb.set_trace()

    # get combine label
    label_num = 0
    combine_label = np.ones_like(rg_label, dtype=np.int32)
    combine_label = -1 * combine_label
    uni_label = np.unique(rgrgb_label)
    uni_label = uni_label[uni_label.argsort()]
    # print(uni_label)
    # pdb.set_trace()
    for idx in range(uni_label.shape[0]):
        if uni_label[idx] <= -1:
            continue
        sp_mask = np.equal(rgrgb_label, uni_label[idx])
        rg_in_sp = rg_label[sp_mask]
        rg_in_sp_tmp = rg_in_sp.copy()
        uni_rg_in_sp = np.unique(rg_in_sp)
        for k in range(uni_rg_in_sp.shape[0]):
            if uni_rg_in_sp[k] <= -1:
                continue
            rg_in_sp_tmp[rg_in_sp == uni_rg_in_sp[k]] = label_num
            label_num += 1
        combine_label[sp_mask] = rg_in_sp_tmp
    print(uni_label.shape[0])
    print(label_num)
    # pdb.set_trace()

    # save txt
    np.savetxt(join(save_path, rg_file), combine_label, fmt="%d", delimiter="\n")
    # pdb.set_trace()

    # save ply
    ply_file = join(ply_path, rg_file[:-4] + '.ply')
    data = read_ply(ply_file)
    ins_colors = random_colors(len(np.unique(combine_label)) + 1, seed=123)
    uni_label = np.unique(combine_label)
    # print(uni_index)
    Y_colors = np.zeros((combine_label.shape[0], 3), np.uint8)
    print(Y_colors.shape)
    for id, semins in enumerate(uni_label):
        valid_ind = np.argwhere(combine_label == semins)[:, 0]
        if semins <= -1:
            tp = [0, 0, 0]
        else:
            tp = np.array(ins_colors[id], dtype=np.float32)
            tp[0] = 255 * tp[0]
            tp[1] = 255 * tp[1]
            tp[2] = 255 * tp[2]
            # print(tp)
        Y_colors[valid_ind] = tp
    points = np.vstack((data['x'], data['y'], data['z'])).T
    _, ply_file_name = os.path.split(ply_file)
    output_filename = join(save_ply_path, ply_file_name)
    print(output_filename)
    write_ply(output_filename, [points, Y_colors],
              ['x', 'y', 'z', 'red', 'green', 'blue'], triangular_faces=None)
    # pdb.set_trace()
