import datetime
import functools
import glob
import itertools as IT
import math
import os
import pickle
import shutil
from os.path import basename, dirname

import cv2
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.misc
from keras.engine import Model
from keras.models import load_model
from keras.preprocessing.image import random_channel_shift
from sklearn.utils import shuffle
from tqdm import tqdm

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

DEBUG = False
DATA_DIR = "data"
OUTPUT_DIR = "output"
ARCHIVE_DIR = "archive"
DATA_GENERATED_DIR = os.path.join(DATA_DIR, "generated")
LOG_FILE = os.path.join(DATA_DIR, "driving_log.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "model-{epoch:02d}-{val_loss:.2f}.h5")
MODEL_HISTORY_FILE = os.path.join(OUTPUT_DIR, "model_history_history.p")
ARGS_FILE = os.path.join(OUTPUT_DIR, "args.txt")


def save_args(args):
    with open(ARGS_FILE, "w") as f:
        txt = ""
        for key, value in args.items():
            txt += '{:>20}: {:<20}\n'.format(key, value)
        print(txt, file=f)


def post_init():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_GENERATED_DIR):
        os.makedirs(DATA_GENERATED_DIR, exist_ok=True)


def load_image(filepath, rel_path=None):
    if rel_path is not None:
        return mpimg.imread(os.path.join(rel_path, filepath))

    return mpimg.imread(filepath)


def save_image(filepath, image, angle=None, debug=DEBUG):
    if debug:
        scipy.misc.toimage(update_image_with_guides(image, angle)).save(filepath)
    else:
        scipy.misc.toimage(image).save(filepath)


def load_images(filepaths, rel_path=None):
    images = []
    for file in filepaths:
        image = load_image(file, rel_path)
        images.append(image)

    return np.array(images)


def load_driving_log(logfile=LOG_FILE, x_cols=None, y_cols=None, size=-1, random=False):
    if x_cols is None:
        x_cols = ['center', 'left', 'right']

    if y_cols is None:
        y_cols = ['steering']

    data_frame = pd.read_csv(logfile)
    X, y = data_frame[x_cols].values.squeeze(), data_frame[y_cols].values.squeeze()

    if random:
        X, y = shuffle(X, y)

    if size > 0:
        X, y = X[:size], y[:size]

    return X, y


def save_driving_log(driving_log, X, y, cols=None):
    if cols is None:
        cols = ['center', 'steering']

    data = np.hstack((X[:, None], y[:, None]))
    df = pd.DataFrame(columns=cols, data=data)
    df.to_csv(driving_log, sep=',', index=False)


def update_image_with_guides(image, angle, pred_angle=None):
    updated_image = image.copy()
    h, w = updated_image.shape[0:2]
    cv2.putText(updated_image, str(round(angle, 2)), org=(2, 33), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5, color=(0, 255, 0), thickness=2)
    cv2.line(updated_image, (int(w / 2), int(h)), (int(w / 2 + angle * w / 4), int(h / 2)), (0, 255, 0),
             thickness=2)
    if pred_angle is not None:
        cv2.putText(updated_image, str(round(pred_angle, 2)), org=(2, 55), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5, color=(0, 0, 255), thickness=2)
        cv2.line(updated_image, (int(w / 2), int(h)), (int(w / 2 + pred_angle * w / 4), int(h / 2)), (0, 0, 255),
                 thickness=2)

    return updated_image


def generate_dirname(original_dir, fileprefix=None):
    if fileprefix is None:
        fileprefix = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S_%f')

    dirname = basename(original_dir)
    return "%s_%s" % (dirname, fileprefix)


def archive_state(archive_name=None):
    if archive_name is None:
        archive_name = generate_dirname(ARCHIVE_DIR)

    target_archive_dir = os.path.join(ARCHIVE_DIR, archive_name)
    target_data_dir = os.path.join(target_archive_dir, DATA_DIR)

    if not os.path.exists(target_archive_dir):
        os.makedirs(target_archive_dir, exist_ok=True)

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir, exist_ok=True)

    if os.path.exists(OUTPUT_DIR):
        shutil.move(OUTPUT_DIR, target_archive_dir)
        print("{} -> {}/".format(OUTPUT_DIR, target_archive_dir), flush=True)

    if os.path.exists(DATA_GENERATED_DIR):
        shutil.move(DATA_GENERATED_DIR, target_data_dir)
        print("{} -> {}/".format(DATA_GENERATED_DIR, target_archive_dir), flush=True)


def brightness_shift(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    bright_increase = int(30 * np.random.uniform(-0.3, 1))
    image_hsv[:, :, 2] = image[:, :, 2] + bright_increase

    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image


def channel_shift(image, intensity=30, channel_axis=2):
    if np.random.rand() > 0.5:
        image = np.invert(image)
    return random_channel_shift(image, intensity, channel_axis)


def flip(image, angle):
    if math.isclose(0.0, angle, rel_tol=1e-5):
        return np.fliplr(image), angle

    return np.fliplr(image), -angle


def random_transform(image, angle):
    image, angle = flip(image, angle)
    image = brightness_shift(image)
    image = channel_shift(image)
    return image, angle


def flatten_driving_log(image_paths, angles, correction, rel_in_path=DATA_DIR):
    res_images_paths = []
    res_angles = []

    for image_path, angle in tqdm(zip(image_paths, angles)):
        (center, left, right) = image_path

        center, left, right = os.path.join(rel_in_path, center.strip()), \
                              os.path.join(rel_in_path, left.strip()), \
                              os.path.join(rel_in_path, right.strip())

        center_angle = angle
        left_angle = angle + correction
        right_angle = angle - correction

        res_images_paths.append(center)
        res_images_paths.append(left)
        res_images_paths.append(right)

        # if center camera image -> no correction
        res_angles.append(center_angle)
        # if left camera image -> correct steering to the right
        res_angles.append(left_angle)
        # if right camera image -> correct steering to the left
        res_angles.append(right_angle)

    return np.array(res_images_paths), np.array(res_angles)


def augment_driving_log(image_paths, angles, out_path=DATA_GENERATED_DIR):
    res_images_paths = []
    res_angles = []

    for image_path, angle in tqdm(zip(image_paths, angles)):
        transformed_image, transformed_angle = \
            random_transform(load_image(image_path), angle)
        transformed_path = os.path.join(out_path, basename(image_path))
        save_image(transformed_path, transformed_image, transformed_angle)
        res_images_paths.append(transformed_path)
        res_angles.append(transformed_angle)

    return np.array(res_images_paths), np.array(res_angles)


def calc_optimal_distribution(angles, bins_num):
    _, bins = np.histogram(angles, bins_num)
    mean = np.mean(angles)
    variance = np.var(angles)
    sigma = np.sqrt(variance)
    x = np.linspace(min(angles), max(angles), bins_num)
    dx = bins[1] - bins[0]
    scale = len(angles) * dx
    y = mlab.normpdf(x, mean, sigma) * scale
    return bins, y


def reduce_size_of(range_x, range_y, over_limit_num):
    data_size = len(range_x)
    target_size = data_size - over_limit_num
    selected_indices = np.random.choice(data_size, size=target_size, replace=False)

    return range_x[selected_indices], range_y[selected_indices]


def adjust_with_best_fit(X_input, y_input, angle_groups):
    bins, y = calc_optimal_distribution(y_input, bins_num=angle_groups)
    X_adj = y_adj = None
    for idx, range_bin in tqdm(enumerate(bins)):
        if idx + 1 == len(bins):
            break

        range_from = range_bin
        range_to = bins[idx + 1]
        optimal_sample_num = int(y[idx])

        if optimal_sample_num == 0:
            continue

        y_indexes = np.sort(np.where((y_input >= range_from) & (y_input < range_to))[0])

        range_x = X_input[y_indexes]
        range_y = y_input[y_indexes]
        data_size = len(range_y)
        assert data_size == len(y_indexes) == len(range_x) == len(range_y)

        # print("dx: ", (range_to - range_from))
        # print("{} ... {},  {} -> {}".format(range_from, range_to, data_size, optimal_sample_num))

        if data_size == 0:
            continue

        missing_samples = optimal_sample_num - data_size
        # print("{} opt. - {} act. = {} est.".format(optimal_sample_num, data_size, missing_samples))

        new_range_x = new_range_y = None
        if missing_samples < 0:
            over_limit_num = abs(missing_samples)
            # print("over {}".format(over_limit_num))
            new_range_x, new_range_y = reduce_size_of(range_x, range_y, over_limit_num)
            # items reduced
            assert len(new_range_x) == len(new_range_y) == optimal_sample_num
        else:
            # print("ok")
            new_range_x, new_range_y = range_x, range_y

        # make one default value for 1 bin range - average
        optimal_value = np.mean(new_range_y)
        # print("Optimal value: ", optimal_value)
        # simply extend array with optimal value
        new_range_y = np.repeat(optimal_value, len(new_range_y))

        assert new_range_x.shape == new_range_y.shape

        if X_adj is None or y_adj is None:
            X_adj, y_adj = new_range_x, new_range_y
        else:
            X_adj = np.concatenate((X_adj, new_range_x), axis=0)
            y_adj = np.concatenate((y_adj, new_range_y), axis=0)

    return X_adj, y_adj


def load_input_data(correction, angle_groups,
                    init_size=-1, init_shuffle=False, logfile=LOG_FILE):
    print("Load data", flush=True)
    X, y = load_driving_log(logfile=logfile, size=init_size, random=init_shuffle)

    print("Flatten data", flush=True)
    X_flat, y_flat = flatten_driving_log(X, y, correction=correction)

    print("Augment data", flush=True)
    X_input, y_input = augment_driving_log(X_flat, y_flat)

    print("Adjust data", flush=True)
    X_adj, y_adj = adjust_with_best_fit(X_input, y_input, angle_groups)

    return X_adj, y_adj


def model_history_available(model_history_file_path=MODEL_HISTORY_FILE):
    return os.path.exists(model_history_file_path)


def load_model_history(model_history_file_path=MODEL_HISTORY_FILE):
    with open(model_history_file_path, 'rb') as f:
        return pickle.load(f)


def save_model_history(history, model_history_file_path=MODEL_HISTORY_FILE):
    if history is None:
        return

    with open(model_history_file_path, 'wb') as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)


def plot_images(images, angles=None, cols=None, squeeze=True, title=None, savepath=None):
    if cols is None:
        cols = len(images)

    rows = len(images) // cols

    if squeeze:
        images = images.squeeze()

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.subplots_adjust(top=0.8)

    if cols == rows == 1:
        axs = [axs]
    else:
        axs = axs.ravel()

    for row in range(rows):
        for col in range(cols):
            index = col + row * cols
            image = images[index]
            axs[index].axis('off')
            if len(image.shape) == 2:
                axs[index].imshow(image, cmap="gray")
            else:
                axs[index].imshow(image)

            if angles is not None:
                axs[index].set_title(angles[index])
    if title is not None:
        plt.suptitle(title, fontsize=12)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_distribution(angles, bins_num, range=(-1, 1), title=None, savepath=None):
    plt.figure(figsize=(12, 6))

    plt.xlim(range)

    if title is not None:
        plt.title('Distribution for {}'.format(title))
    plt.hist(angles, bins=bins_num)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_optimal_distribution(angles, bins_num, range=(-1, 1)):
    plt.figure(1, figsize=(12, 6))

    plt.hist(angles, bins_num)

    plt.xlim(range)
    x = np.linspace(min(angles), max(angles), bins_num)

    _, y = calc_optimal_distribution(angles, bins_num)

    plt.scatter(x, y, marker='+', s=1, c='r', lw=6, zorder=10)

    plt.tight_layout()
    plt.show()


def plot_model_history(filepath, title, lbls, show=True, ylim=(0, 0.05), xlim=(0, 30)):
    if model_history_available(model_history_file_path=filepath):
        history = load_model_history(model_history_file_path=filepath)
        if show:
            plt.figure(figsize=(8, 6))

        min_results = []
        for lbl in lbls:
            if show:
                plt.plot(history['epoch'], history[lbl], label=lbl)
            min_value_idx = np.argmin(history[lbl])
            epoch_number = history['epoch'][min_value_idx]
            min_value = history[lbl][min_value_idx]
            min_results.append((lbl, epoch_number, min_value))

        if show:
            plt.title(title)
            plt.ylabel('Data')
            plt.xlabel('Epochs')
            plt.legend(loc='lower right')
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.tight_layout()
            plt.show()
        return min_results
    else:
        print("{} not found".format(filepath))
        return None


def plot_model_history_list(histories, lbls=('loss', 'val_loss', 'mean_squared_error'), plot_func=plot_model_history,
                            show=True, ylim=(0, 0.05), xlim=(0, 30)):
    aggr_results = []
    for path, name in histories:
        results = plot_func(path, name, lbls, show, ylim, xlim)
        aggr_results.append((name, results))
    return aggr_results


def plot_images_with_ids(x, y, image_ids, rel_path=None, grid_cols=4):
    rnd_images = load_images(x[image_ids], rel_path)
    rnd_angles = y[image_ids]

    rnd_mod_images = []
    rnd_mod_classes = []
    for image, clazz in zip(rnd_images, rnd_angles):
        rnd_mod_images.append(update_image_with_guides(image, clazz))
        rnd_mod_classes.append(clazz)

    rnd_mod_images = np.array(rnd_mod_images)
    rnd_mod_classes = np.array(rnd_mod_classes)

    plot_images(rnd_mod_images, rnd_mod_classes, cols=grid_cols)

    print(rnd_mod_images.shape)
    print(rnd_mod_classes.shape)


def plot_random_images(x, y, rel_path=None, size=8, grid_cols=4):
    selected_ids = np.random.choice(len(x), size=size, replace=False)
    plot_images_with_ids(x, y, selected_ids, rel_path, grid_cols)


def save_model_architecture(model_path, output_path='model.png'):
    from keras.utils.visualize_util import plot
    model = load_model(model_path)
    plot(model, to_file=output_path, show_shapes=True)


def get_top_smallest_k_history_metrics(histories, top_k_smallest=5):
    results = plot_model_history_list(histories, show=False)

    loss_items = np.array([[item[1][0][1], item[1][0][2]] for item in results])
    val_loss_items = np.array([[item[1][1][1], item[1][1][2]] for item in results])
    mse_items = np.array([[item[1][2][1], item[1][2][2]] for item in results])

    # data[np.argsort(data[:, 0])] where the 0 is the column index on which to sort
    loss_items_top_k_sorted_indexes = np.argsort(loss_items[:, 1])[:top_k_smallest]
    val_loss_items_top_k_sorted_indexes = np.argsort(val_loss_items[:, 1])[:top_k_smallest]
    mse_items_top_k_sorted_indexes = np.argsort(mse_items[:, 1])[:top_k_smallest]

    top_k_loss_items = [(results[idx][0], loss_items[idx].tolist()) for idx in
                        loss_items_top_k_sorted_indexes]
    top_k_val_loss = [(results[idx][0], val_loss_items[idx].tolist()) for idx in
                      val_loss_items_top_k_sorted_indexes]
    top_k_mse = [(results[idx][0], mse_items[idx].tolist()) for idx in mse_items_top_k_sorted_indexes]

    return top_k_loss_items, top_k_val_loss, top_k_mse


def concat_n_images(images, rows, cols, cellHeight, cellWidth, target_depth=1):
    display = np.empty((cellHeight * rows, cellWidth * cols, target_depth), dtype=np.float32)

    for i, j in IT.product(range(rows), range(cols)):
        # arr = cellArray[i*ncols+j].image  # you may need this
        arr = images[i * cols + j]  # my simplified cellArray uses this
        x, y = i * cellHeight, j * cellWidth
        display[x:x + cellHeight, y:y + cellWidth, :] = arr
    return display


def conv_layer_output(model, layer, image):
    depth = layer.output_shape[-1]
    image = np.expand_dims(image, axis=0)

    test_model = Model(input=model.input, output=layer.output)

    conv_features = test_model.predict(image)
    # print("Convolutional features shape: ", conv_features.shape)

    feature_list = []
    for i in range(depth):
        conv_feature = conv_features[0, :, :, i]
        conv_feature = np.expand_dims(conv_feature, axis=len(conv_feature.shape))
        feature_list.append(conv_feature)

    return np.array(feature_list)


def combine_images_vert(a, b):
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    total_height = ha + hb
    max_width = np.max([wa, wb])
    new_img = np.zeros(shape=(total_height, max_width, 1), dtype=np.float32)

    new_img[:ha, :wa] = a
    new_img[ha:ha + hb, :wb] = b

    return new_img


def combine_images_horiz(a, b):
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 1), dtype=np.float32)

    new_img[:ha, :wa] = a
    new_img[:hb, wa:wa + wb] = b

    return new_img


def put_text(image, text, color=(255, 255, 255), make_border=True):
    image = image.copy()
    if np.max(image) <= 1:
        image = ((image + 0.5) * 255).round().astype(np.uint8)

    image = cv2.putText(img=image, text=text, org=(2, 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=color, thickness=1)

    if make_border:
        image = cv2.copyMakeBorder(image.squeeze(), 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255))
        image = np.expand_dims(image, axis=len(image.shape))
    return image


def save_image_feature_map(model, conv_layers, test_image, grid_setup, filepath):
    outdir = dirname(filepath)

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    feature_map_images = []
    for layer in conv_layers:
        layer_height, layer_width, layer_depth = layer.output_shape[1:]

        cell_height, cell_width = layer_height, layer_width
        rows, cols = grid_setup[layer_depth]

        normalized_test_image = test_image.copy()
        layer_output_image = conv_layer_output(model, layer, normalized_test_image)

        concat_images = concat_n_images(layer_output_image, rows, cols, cell_height, cell_width, 1)
        title = layer.name
        concat_images = put_text(concat_images, title, make_border=True)

        feature_map_images.append(concat_images)

    merged_image = functools.reduce(combine_images_vert, feature_map_images)

    plt.figure(figsize=(16, 10))
    plt.axis('off')
    plt.imshow(merged_image.squeeze(), cmap="gray")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')


def save_all_images_feature_map(image_dir, model_path, rel_path=None, target_dir=DATA_GENERATED_DIR):
    grid_setup = {
        3: (1, 3),
        24: (4, 6),
        36: (6, 6),
        48: (6, 8),
        64: (8, 8)
    }

    X = np.array(sorted([path for path in glob.iglob(os.path.join(image_dir, "*.jpg"), recursive=True)]))

    model = load_model(model_path)
    model.summary()

    # we are interesting in first 5 layers, since next layers are more abstract and small to view anything
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # ->  lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
    # ____________________________________________________________________________________________________
    # ->  cropping2d_1 (Cropping2D)        (None, 66, 320, 3)    0           lambda_1[0][0]
    # ____________________________________________________________________________________________________
    # ->  convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
    # ____________________________________________________________________________________________________
    # ->  convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
    # ____________________________________________________________________________________________________
    # ->  convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
    # ____________________________________________________________________________________________________
    # convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
    # ____________________________________________________________________________________________________
    # convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
    # ____________________________________________________________________________________________________
    # dropout_1 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5[0][0]
    # ____________________________________________________________________________________________________
    # flatten_1 (Flatten)              (None, 2112)          0           dropout_1[0][0]
    # ____________________________________________________________________________________________________
    # dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
    # ____________________________________________________________________________________________________
    # dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
    # ____________________________________________________________________________________________________
    # dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
    # ____________________________________________________________________________________________________
    # dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
    # ====================================================================================================
    selected_layers = model.layers[0:-8]

    for original_file in tqdm(X):
        test_image = load_image(original_file, rel_path=rel_path)
        test_image_file_path = os.path.join(target_dir, basename(original_file))
        save_image_feature_map(model, selected_layers, test_image, grid_setup, test_image_file_path)


if __name__ == "__main__":
    archive_state()
    post_init()

    IMAGE_DIR = "/Users/volkodav/self_driving_car_proj/proj/CarND-Behavioral-Cloning-P3/video"
    MODEL_PATH = "/Users/volkodav/self_driving_car_proj/proj/work9/relu_nogeneration_rnd_zoom_brightness_channel_shift_cor010/output/model-09-0.01.h5"
    save_all_images_feature_map(IMAGE_DIR, MODEL_PATH, target_dir="output_featuremap")
    # work9/relu_nogeneration_rnd_zoom_brightness_channel_shift_cor010/output/model-09-0.01.h5

    save_model_architecture(MODEL_PATH)
