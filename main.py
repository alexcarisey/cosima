import os
import sys
from time import time
import re
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import mpl_interactions.ipyplot as iplt
from PIL import Image
import shutil

pl.Config.set_fmt_str_lengths(1000)
pl.Config.set_tbl_rows(1000)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)

IMG_FORMAT = {".nd2": 0,
              ".tif": 0
              }

TABLE_FORMAT = {".csv": 0}

ILASTIK_COLS = ['object_id',
                'Predicted Class',
                'Probability of LipidDroplets',
                'Object Center_0',
                'Object Center_1',
                'Object Area',
                'Radii of the object_0',
                'Radii of the object_1',
                'Size in pixels',
                'Bounding Box Maximum_0',
                'Bounding Box Maximum_1',
                'Bounding Box Minimum_0',
                'Bounding Box Minimum_1',
                'Diameter']

# lazy load csv
# arr_cen = pl.scan_csv(r'./raw_table/C1-ER-LD+3uM DOM 60x confocal Z003.nd2_10006_table.csv') \
#     .select(ILASTIK_COLS) \
#     .filter(pl.col("Predicted Class") == "LipidDroplets")
# arr_cen = arr_cen.collect()
#
# print(arr_cen, arr_cen.shape[0])

RING_THICKNESS = 5
WALKER_MAX = 253


def time_elapsed(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def file_ext_check(input_path, img_format):
    print("checking format...")
    print(img_format)

    if not os.path.exists(bmask_path) or not os.path.exists(bmask_table_path):
        print("missing necessary folder...")
        os.makedirs(bmask_path)
        os.makedirs(bmask_table_path)

    file_list = os.listdir(input_path)
    check_list = [None] * len(file_list)

    for i_file, row_file in enumerate(file_list):
        # Split the extension from the path and make it lowercase.
        name = os.path.splitext(row_file)[0]
        # .os.path.splitext(row_file)[0]
        ext = os.path.splitext(row_file)[-1].lower()
        print(i_file, name, ext)
        if ext == ".tif":
            if re.search("_Object Predictions", name):
                print("src: ", os.path.realpath(input_path + '/' + row_file))
                print("src: ", os.path.realpath(bmask_table_path + '/' + row_file))
                os.replace(os.path.realpath(input_path + '/' + row_file), os.path.realpath(bmask_path + '/' + row_file))
            else:
                img_format[ext] += 1
                check_list[i_file] = True  # not useful here
                file_list[i_file] = name
        if ext == ".csv":
            shutil.move(os.path.realpath(input_path + '/' + row_file), os.path.realpath(bmask_table_path + '/' + row_file))

    file_table = pl.DataFrame({"file": file_list, "droplet": check_list}).filter(pl.col('droplet'))
    return file_table


# @time_elapsed
# def load_image(file, path):
#     #     files = file_ext_check(input_path, img_format)
#     print("loading images...")
#     print(os.path.realpath(path) + '/' + file)
#     dataset = ij.io().open(os.path.realpath(path) + '/' + file)
#     #     print(type(dataset),dataset)
#     print(type(dataset), dataset)
#     print("loading complete")
#     return dataset

# ij.ui().show(dataset)
# try:
#     for f in files:
#         dataset = ij.io().open(os.path.realpath(input_path+"/"+f))
#         ij.py.show(dataset)
# except TypeError:
#     print(TypeError)
# dataset = ij.io().open(os.path.realpath(input_path + "/" + files[0]))
# ij.ui().show(dataset)


def dump_info(image):
    """A handy function to print details of an image object."""
    name = image.name if hasattr(image, 'name') else None  # xarray
    if name is None and hasattr(image, 'getName'): name = image.getName()  # Dataset
    if name is None and hasattr(image, 'getTitle'): name = image.getTitle()  # ImagePlus
    print(f" name: {name or 'N/A'}")
    print(f" type: {type(image)}")
    print(f"dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
    print(f"shape: {image.shape}")
    print(f" dims: {image.dims if hasattr(image, 'dims') else 'N/A'}")


def rm_xy_dup(y, x):
    print(len(x), len(y))
    cir = pl.DataFrame({"x": x, "y": y})

    cir = cir[['x', 'y']].unique()

    # for row in cir.iter_rows(named=True):
    #     print(row['x'], row['y'])
    print(len(cir))
    return cir


def reorder_xy(cir, center, r):
    print(center)
    cir4 = cir.filter((pl.col('y') < center[1]) & (pl.col('x') > center[0]))
    cir4 = cir4.sort('y', 'x', descending=True)

    def cir4_to_3_mirror(row, col_name):
        return 2 * center[0] - row[col_name]

    cir3_x = cir4.pipe(cir4_to_3_mirror, col_name='x')
    cir3 = pl.DataFrame({'x': cir3_x, 'y': cir4['y']})
    cir_down = pl.concat([cir4, pl.DataFrame({'x': [center[0]], 'y': [center[1] - r]}), cir3.reverse()])
    print(cir3)
    print(cir4)
    print(cir_down)
    cir_up_y = cir_down.select('y').pipe(
        lambda row: 2 * center[1] - row['y'])
    print(cir_up_y)
    cir_up = pl.DataFrame({'x': cir_down.get_column('x'), 'y': cir_up_y})
    cir_ord = pl.concat([pl.DataFrame({'x': [center[0] + r], 'y': [center[1]]}),
                         cir_down,
                         pl.DataFrame({'x': [center[0] - r], 'y': [center[1]]}),
                         cir_up.reverse()
                         ])
    if len(cir) == len(cir_ord):
        return cir_ord
    else:
        return cir


def gen_outline(y, x, bmap, void_i):
    if bmap[y, x] == 1:
        bmap[y, x] = 2
        action_on_neighbors(y, x, True, gen_outline, bmap, void_i)

    if bmap[y, x] == 0:
        bmap[y, x] = 255 + void_i * 2


@time_elapsed
def gen_outline_bmask(bmask):
    bmask[bmask == 255] = 1
    for void_i, void in enumerate(arr_cen.iter_rows(named=True)):
        void_x = int(round(void['Object Center_0'], 0))
        void_y = int(round(void['Object Center_1'], 0))
        gen_outline(void_y, void_x, bmask, void_i)

    # plt.imshow(bmask)
    # plt.show()
    # bmask[bmask == 2] = 255
    bmask[bmask == 2] = 0


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If computer is running windows use cls
        command = 'cls'
    os.system(command)


def action_on_neighbors(bmask_y, bmask_x, eight, func, *arg, **kwargs):
    func(bmask_y + 1, bmask_x, *arg, **kwargs)
    func(bmask_y - 1, bmask_x, *arg, **kwargs)
    func(bmask_y, bmask_x + 1, *arg, **kwargs)
    func(bmask_y, bmask_x - 1, *arg, **kwargs)
    if eight:
        func(bmask_y + 1, bmask_x + 1, *arg, **kwargs)
        func(bmask_y - 1, bmask_x - 1, *arg, **kwargs)
        func(bmask_y + 1, bmask_x - 1, *arg, **kwargs)
        func(bmask_y - 1, bmask_x + 1, *arg, **kwargs)


def casting(bmask_y, bmask_x, bmask, cast_num):
    if bmask[bmask_y, bmask_x] == 0:
        bmask[bmask_y, bmask_x] = cast_num


# def detect_collide_ends(bmask_y, bmask_x, bmask, cast_num):


def detect_collide(bmask_y, bmask_x, bmask, edge_num, cast_num):
    if bmask[bmask_y, bmask_x] != cast_num and bmask[bmask_y, bmask_x] != edge_num and bmask[bmask_y, bmask_x] > 255:
        # print("colllide:", y_edge + 1, x_edge, b_mask[y_edge + 1, x_edge])
        overlap_xy.append([bmask_y, bmask_x])


def edge_recur(y_edge, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output, branch_data,
               count, last_dir=None):
    # os.system('cls')
    # time.sleep(1)
    # input("press enter to continue...")
    # clearConsole()

    original_num = b_mask[y_edge, x_edge]
    print(y_edge, x_edge, original_num)
    # casting
    if b_mask[y_edge, x_edge] == edge_num:
        action_on_neighbors(y_edge, x_edge, True, casting, b_mask, cast_num)
        # action_on_neighbors(y_edge, x_edge, True, detect_collide, b_mask, edge_num, cast_num)

    if  b_mask[y_edge, x_edge] == 254:
        # b_mask[y_edge, x_edge] = 255
        overlap_xy.append([y_edge, x_edge])
        record_ring(y_edge, x_edge, ring_output, branch_data, ring_map[y_edge, x_edge], contact_map[y_edge, x_edge],
                    count,
                    True)
    else:
        # b_mask[y_edge, x_edge] = (count % WALKER_MAX + 1)
        record_ring(y_edge, x_edge, ring_output, branch_data, ring_map[y_edge, x_edge], contact_map[y_edge, x_edge],
                    count,
                    False)

    b_mask[y_edge, x_edge] = (count % WALKER_MAX + 1)
    count += 1
    # 3rd quadrant
    if y_edge > yc and x_edge <= xc:
        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or ( b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='r')

        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or ( b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='d')

        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or ( b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='u')

        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or ( b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='l')

    # 4th quadrant
    if x_edge > xc and y_edge >= yc:
        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or ( b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='u')

        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or ( b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='r')

        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or ( b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='l')

        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or ( b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='d')

    # 1st quadrant
    if y_edge < yc and x_edge >= xc:
        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or ( b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='l')

        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or ( b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='u')

        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or ( b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='r')

        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or ( b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='d')

    # 2nd quadrant
    if x_edge < xc and y_edge <= yc:
        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or ( b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='d')

        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or ( b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='l')

        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or ( b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='u')

        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or ( b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, last_dir='r')

    # # collide edge
    if original_num == 254:
        if b_mask[y_edge + 1, x_edge] != cast_num and b_mask[y_edge - 1, x_edge] != cast_num and b_mask[
            y_edge, x_edge + 1] != cast_num and b_mask[y_edge, x_edge - 1] != cast_num \
                and (b_mask[y_edge + 1, x_edge] == 254 or b_mask[y_edge - 1, x_edge] == 254 or b_mask[
            y_edge, x_edge + 1] == 254 or b_mask[y_edge, x_edge - 1] == 254):
            action_on_neighbors(y_edge, x_edge, False, detect_collide, b_mask, edge_num, cast_num)
    else:
        if b_mask[y_edge + 1, x_edge] != 254 and b_mask[y_edge - 1, x_edge] != 254 and b_mask[
            y_edge, x_edge + 1] != 254 and b_mask[y_edge, x_edge - 1] != 254:
            action_on_neighbors(y_edge, x_edge, True, detect_collide, b_mask, edge_num, cast_num)
    # if b_mask[y_edge, x_edge] < 254:
    #     action_on_neighbors(y_edge, x_edge, True, detect_collide, b_mask, cast_num)


def rm_inner_casting(y,x,bmask, target):
    print("removing...", target)
    if bmask[y,x] == target:
        bmask[y,x] = 255
        action_on_neighbors(y, x, False, rm_inner_casting, bmask, target)


def record_ring(y_edge, x_edge, ring_output, branch_output, ring_inten, contact_inten, count, overlap_flag):
    if ring_output['ring_y'][count] != 0 and ring_output['ring_x'][count] != 0:
        print("branch detected!!!!!")
        print(ring_output['ring_y'][count], ring_output['ring_x'][count], ring_output['index'][count])
        print(y_edge, x_edge, count)
        branch_output['index'][count] = count
        branch_output['ring_y'][count] = y_edge
        branch_output['ring_x'][count] = x_edge
        branch_output['ring_inten'][count] = ring_inten
        branch_output['contact_inten'][count] = contact_inten
        if overlap_flag is True:
            branch_output['overlap'][count] = True
    else:
        ring_output['index'][count] = count
        ring_output['ring_y'][count] = y_edge
        ring_output['ring_x'][count] = x_edge
        ring_output['ring_inten'][count] = ring_inten
        ring_output['contact_inten'][count] = contact_inten
        if overlap_flag is True:
            ring_output['overlap'][count] = True


def conv(val):
    try:
        return int(val)
    except ValueError:
        return None


def plot_layer(l=0):
    fig_l, ax_l = plt.subplots(2, 2)
    if l < RING_THICKNESS:
        ax_l[0, 0].set_title(f"layer: {l + 1}")
    else:
        ax_l[0, 0].set_title("projection")

    for row_cen in arr_cen.iter_rows(named=True):
        label = f"{row_cen['object_id']}"
        ax.annotate(label,  # this is the text
                    (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                    color='red',
                    # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 0),  # distance from text to points (x,y)
                    ha='center')  # horizontal alignment can be left, right or center
        # ax.text(int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0)),
        #          f"{row_cen['object_id']}", **text_kwargs)
    ax.imshow(map_layer[l])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # change mode to headless for cluster environment
    # ij = imagej.init('sc.fiji:fiji:2.10.0', mode="interactive")
    # print(ij.getVersion())
    #
    # # HACK: # HACK: Work around ImagePlus#show() failure if no ImagePlus objects are already registered.
    # if ij.WindowManager.getIDList() is None:
    #     ij.py.run_macro('newImage("dummy", "8-bit", 1, 1, 1);')

    # check first time runner
    bmask_path = os.path.realpath("./bmask")
    bmask_table_path = os.path.realpath("./bmask_table")
    droplet_path = os.path.realpath("./droplet")
    contact_path = os.path.realpath("./contact")
    print(bmask_path, bmask_table_path, droplet_path, contact_path)

    # check format in batch
    file_lookup_list = file_ext_check(droplet_path, IMG_FORMAT)
    print(file_lookup_list)

    # load images and tables from file_lookup_list
    for item in file_lookup_list.iter_rows(named=True):
        if item['droplet'] is True:
            # lazy load centroids table
            print(os.path.realpath(bmask_table_path + '/' + item['file'] + "_table.csv"))
            arr_cen = pl.scan_csv(os.path.realpath(bmask_table_path + '/' + item['file'] + "_table.csv")) \
                .select(ILASTIK_COLS) \
                .filter(pl.col("Predicted Class") == "LipidDroplets")
            arr_cen = arr_cen.collect()
            print(arr_cen, arr_cen.shape)

            # form outline from bmask
            # print()
            map_bmask = Image.open(os.path.realpath(bmask_path + '/' + item['file'] + "_Object Predictions.tif"))
            arr_bmask = np.array(map_bmask).astype(np.uint16, casting="same_kind")
            edge_copy = arr_bmask.copy()
            gen_outline_bmask(edge_copy)

            plt.imshow(edge_copy)
            plt.show()

            # load droplet intensity
            map_droplet = Image.open(os.path.realpath(droplet_path + '/' + item['file'] + ".tif"))
            arr_droplet = np.array(map_droplet)
            plt.imshow(arr_droplet)
            plt.show()

            # load contact intensity
            # print(os.path.realpath(contact_path + '/' + item['file'] + ".tif"))
            map_contact = Image.open(os.path.realpath(contact_path + '/' + item['file'].replace(" C=0_", " C=1_") + ".tif"))
            arr_contact = np.array(map_contact)
            plt.imshow(arr_contact)
            plt.show()

            t = 0
            ring_datatable = None
            branch_datatable = None
            map_layer = np.full((RING_THICKNESS + 1, np.shape(edge_copy)[0], np.shape(edge_copy)[1]), 0)

            while t < RING_THICKNESS:
                for i, row in enumerate(arr_cen.iter_rows(named=True)):
                    print("i:", i)
                    # allocate table for ring data
                    ring_len = int(row['Size in pixels']) * 10
                    print(row['object_id'], ring_len)
                    overlap_xy = []

                    ring_data = {
                        'object_id': np.full((ring_len,), row['object_id'], dtype=np.uint16),
                        'layer': np.full((ring_len,), t + 1, dtype=np.uint16),
                        'index': np.zeros((ring_len,), dtype=np.uint16),
                        'ring_y': np.zeros((ring_len,), dtype=np.uint16),
                        'ring_x': np.zeros((ring_len,), dtype=np.uint16),
                        'ring_inten': np.zeros((ring_len,), dtype=np.uint16),
                        'contact_inten': np.zeros((ring_len,), dtype=np.uint16),
                        'overlap': np.full((ring_len,), False, dtype=bool)
                    }

                    branch_data = {
                        'object_id': np.full((ring_len,), row['object_id'], dtype=np.uint16),
                        'layer': np.full((ring_len,), t + 1, dtype=np.uint16),
                        'index': np.zeros((ring_len,), dtype=np.uint16),
                        'ring_y': np.zeros((ring_len,), dtype=np.uint16),
                        'ring_x': np.zeros((ring_len,), dtype=np.uint16),
                        'ring_inten': np.zeros((ring_len,), dtype=np.uint16),
                        'contact_inten': np.zeros((ring_len,), dtype=np.uint16),
                        'overlap': np.full((ring_len,), False, dtype=bool)
                    }

                    # track centroid for each void
                    cen_x = int(round(row['Object Center_0'], 0))
                    cen_y = int(round(row['Object Center_1'], 0))

                    edge = 255 + t + 2 * i
                    cast = edge + 1

                    # if t == 0:
                    #     edge = 255
                    #     cast = edge + 1 + 2 * i
                    # else:
                    #     edge = 255 + t + 2 * i
                    #     cast = edge + 1

                    if t == 0:
                        y_Start = cen_y
                    else:
                        start_last_layer = ring_datatable.filter(
                            (pl.col("object_id") == row['object_id']) & (pl.col("layer") == t) & (pl.col("index") == 0))
                        # print(start_last_layer)
                        y_Start = start_last_layer['ring_y'][0]

                    if t ==1:
                        print('removing inner casting...')
                        rm_inner_casting(y_Start-1,cen_x,edge_copy,edge)

                    print("starting(y, x):", y_Start, cen_x)

                    while not (edge_copy[y_Start, cen_x] == edge or edge_copy[y_Start, cen_x] == 254):
                        y_Start += 1
                        print(edge_copy[y_Start,cen_x], edge, y_Start, cen_x)
                    # print(type(ring_data['layer']))
                    print("void info:", y_Start, cen_x, edge, cast, edge_copy[y_Start, cen_x], row['object_id'])

                    edge_recur(y_Start, cen_x, edge_copy, arr_droplet, arr_contact, edge, cast, cen_y, cen_x, ring_data,
                               branch_data,
                               0)  # y_edge, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output, count, last_dir= None

                    # patch overlapped spots
                    b4_patch = edge_copy.copy()
                    for item in overlap_xy:
                        edge_copy[item[0], item[1]] = 254

                    # cook droplet data per layer and swap branched pixel
                    for i_branch in reversed(range(len(branch_data['index']))):
                        dup_i = branch_data['index'][i_branch]
                        if (-1 <= ring_data['ring_y'][dup_i + 1] - branch_data['ring_y'][i_branch] <= 1) and (
                                -1 <= ring_data['ring_x'][dup_i + 1] - branch_data['ring_x'][i_branch] <= 1):
                            print("swap branch data.....")
                            print(ring_data['ring_y'][dup_i + 1], ring_data['ring_x'][dup_i + 1],
                                  branch_data['ring_y'][i_branch], branch_data['ring_x'][i_branch])
                            temp = [ring_data['ring_y'][dup_i], ring_data['ring_x'][dup_i],
                                    ring_data['ring_inten'][dup_i],
                                    ring_data['contact_inten'][dup_i]]
                            ring_data['ring_y'][dup_i] = branch_data['ring_y'][i_branch]
                            ring_data['ring_x'][dup_i] = branch_data['ring_x'][i_branch]
                            ring_data['ring_inten'][dup_i] = branch_data['ring_inten'][i_branch]
                            ring_data['contact_inten'][dup_i] = branch_data['contact_inten'][i_branch]
                            print("=========================================================================")

                    ring_data = pl.DataFrame(ring_data).filter((pl.col('ring_y') != 0) & (pl.col('ring_y') != 0))
                    branch_data = pl.DataFrame(branch_data).filter((pl.col('ring_y') != 0) & (pl.col('ring_y') != 0))
                    print(ring_data, branch_data)
                    if ring_datatable is None:
                        ring_datatable = ring_data.clone()
                    else:
                        ring_datatable = pl.concat([ring_datatable, ring_data])

                    if branch_datatable is None:
                        branch_datatable = branch_data.clone()
                    else:
                        branch_datatable = pl.concat([branch_datatable, branch_data])

                    # plot bmask
                    if t >= 5:
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.suptitle(f"droplet id: {ring_data['object_id'][0]} , layer: {ring_data['layer'][0]}")

                        ax1.imshow(edge_copy)
                        text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1')
                        for row_cen in arr_cen.iter_rows(named=True):
                            # label = f"{row_cen['object_id']}"
                            ax1.text(int(round(row_cen['Object Center_0'], 0)),
                                     int(round(row_cen['Object Center_1'], 0)),
                                     f"{row_cen['object_id']}", **text_kwargs)
                            # ax1.annotate(label,  # this is the text
                            #              (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),  # these are the coordinates to position the label
                            #              textcoords="offset points",  # how to position the text
                            #              xytext=(0, 0),  # distance from text to points (x,y)
                            #              ha='center')  # horizontal alignment can be left, right or center
                        ax2.imshow(b4_patch)

                        # label = f"{row['object_id']}"
                        ax2.text(cen_x, cen_y, f"{row['object_id']}", **text_kwargs)
                        # ax1.annotate(label,  # this is the text
                        #              (cen_x, cen_y),  # these are the coordinates to position the label
                        #              textcoords="offset points",  # how to position the text
                        #              xytext=(0, 10),  # distance from text to points (x,y)
                        #              ha='center')  # horizontal alignment can be left, right or center
                        #
                        # ax2.annotate(label,  # this is the text
                        #              (cen_x, cen_y),  # these are the coordinates to position the label
                        #              textcoords="offset points",  # how to position the text
                        #              xytext=(0, 10),  # distance from text to points (x,y)
                        #              ha='center')  # horizontal alignment can be left, right or center

                        # plot intensity
                        # ax2.plot(ring_data['ring_inten'], '-bo')
                        #
                        # for ring_i, ring_row in enumerate(ring_data.iter_rows(named=True)):
                        #     label = f"{ring_row['ring_inten']}"
                        #
                        #     ax2.annotate(label,  # this is the text
                        #                  (ring_i, ring_row['ring_inten']),  # these are the coordinates to position the label
                        #                  textcoords="offset points",  # how to position the text
                        #                  xytext=(0, 10),  # distance from text to points (x,y)
                        #                  ha='center')  # horizontal alignment can be left, right or center
                        #
                        # # plt.plot(ring_data['contact_inten'], '-ro')
                        #
                        # ax2.plot(ring_data['contact_inten'], '-ro')
                        #
                        # for con_i, con_row in enumerate(ring_data.iter_rows(named=True)):
                        #     label = f"{con_row['contact_inten']}"
                        #
                        #     ax2.annotate(label,  # this is the text
                        #                  (con_i, con_row['contact_inten']),  # these are the coordinates to position the label
                        #                  textcoords="offset points",  # how to position the text
                        #                  xytext=(0, 10),  # distance from text to points (x,y)
                        #                  ha='center')  # horizontal alignment can be left, right or center
                        # # plot setting
                        # ax2.legend(["edge intensity", "contact signal"], loc="lower right")
                        # # Set common labels
                        # ax2.set_xlabel('distance (px)')
                        # ax2.set_ylabel('raw intensity')
                        # # ax = plt.axes()

                        # function to show the plot
                        plt.show()

                    # new edge
                    # edge = cast

                map_layer[t] = edge_copy
                map_filtered = edge_copy.copy()
                map_filtered[map_filtered < 254] = 0
                map_filtered[map_filtered >= 254] = (t + 1) * 255
                map_layer[-1] += map_filtered

                print(ring_datatable, branch_datatable)
                # increase thickness
                t += 1

                # plt.imshow(edge_c, interpolation='nearest')
                # plt.show()
                # post-casting
            fig, ax_layer = plt.subplots(1, RING_THICKNESS + 1)
            for i, ax in enumerate(ax_layer):
                # fig.suptitle(f"layer: {i+1}")
                if i < RING_THICKNESS:
                    ax.set_title(f"layer: {i + 1}")
                else:
                    ax.set_title("projection")

                text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1')
                for row_cen in arr_cen.iter_rows(named=True):
                    label = f"{row_cen['object_id']}"
                    ax.annotate(label,  # this is the text
                                (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                                color='red',
                                # these are the coordinates to position the label
                                textcoords="offset points",  # how to position the text
                                xytext=(0, 0),  # distance from text to points (x,y)
                                ha='center')  # horizontal alignment can be left, right or center
                    # ax.text(int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0)),
                    #          f"{row_cen['object_id']}", **text_kwargs)
                ax.imshow(map_layer[i])
            plt.show()


            # ipywidgets.interact(plot_layer, l=(0,RING_THICKNESS+1, 1))
            def show_layer(layer):
                return map_layer[layer - 1]


            fig, ax = plt.subplots()
            plt.subplots_adjust(top=0.9)
            for row_cen in arr_cen.iter_rows(named=True):
                label = f"{row_cen['object_id']}"
                ax.annotate(label,  # this is the text
                            (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                            color='red',
                            # these are the coordinates to position the label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 0),  # distance from text to points (x,y)
                            ha='center')  # horizontal alignment can be left, right or center
                # ax.text(int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0)),
                #          f"{row_
            axfreq = plt.axes([0.05, 0.95, 0.1, 0.01])  # right, top, length, width
            slider = Slider(axfreq, label="layer  ", valmin=1, valmax=RING_THICKNESS + 1, valstep=1)
            controls = iplt.imshow(show_layer, layer=slider, ax=ax)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.show()