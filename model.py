import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import helper

SAVE_BEST_ONLY = False
ACTIVATION = 'relu'
# If dropout rate is set to 0.2 (20%) - one in 5 inputs will be randomly excluded from each update cycle
DROPOUT = 0.5
LEARNING_RATE = 1.0e-4
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 512
NB_EPOCH = 20
VERBOSE = 1
ANGLE_CORRECTION = 0.1
INIT_SIZE = 100
INIT_SHUFFLE = False
ARCHIVE_NAME = helper.generate_dirname(helper.ARCHIVE_DIR)
ANGLE_GROUPS = 200

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_boolean('save_best_only', SAVE_BEST_ONLY,
                     "Keep the latest best model according to the quantity monitored or overwrite it. Default: %s" % str(
                         SAVE_BEST_ONLY))
flags.DEFINE_string('activation', ACTIVATION, "Name of activation function to use. Default: %s" % str(ACTIVATION))
flags.DEFINE_float('learning_rate', LEARNING_RATE, "Learning rate. Default: %s" % str(LEARNING_RATE))
flags.DEFINE_float('dropout', DROPOUT, "Fraction of the input units to drop. Default: %s" % str(DROPOUT))
flags.DEFINE_float('validation_split', VALIDATION_SPLIT,
                   "Proportion of the dataset to include in the test. Default: %s" % str(VALIDATION_SPLIT))
flags.DEFINE_integer('batch_size', BATCH_SIZE, "The batch size. Default: %s" % str(BATCH_SIZE))
flags.DEFINE_integer('nb_epoch', NB_EPOCH, "Total number of iterations on the data. Default: %s" % str(NB_EPOCH))
flags.DEFINE_integer('verbose', VERBOSE, "Verbosity mode, 0, 1, or 2. Default: %s" % str(VERBOSE))
flags.DEFINE_float('correction', ANGLE_CORRECTION,
                   "Steering angle adjustment for the side camera images. Default: %s" % str(ANGLE_CORRECTION))
flags.DEFINE_integer('init_size', INIT_SIZE, "Set initial size of input to load. Default: %s" % str(INIT_SIZE))
flags.DEFINE_boolean('init_shuffle', INIT_SHUFFLE,
                     "Shuffle input data. Default: %s" % str(INIT_SHUFFLE))
flags.DEFINE_string('archive_name', ARCHIVE_NAME, "Name of archive. Default: %s" % str(ARCHIVE_NAME))
flags.DEFINE_integer('angle_groups', ANGLE_GROUPS, "Number of angle groups. Default: %s" % str(ANGLE_GROUPS))


def build_model(input_shape=helper.INPUT_SHAPE):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((71, 23), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation=FLAGS.activation))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation=FLAGS.activation))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation=FLAGS.activation))
    model.add(Convolution2D(64, 3, 3, activation=FLAGS.activation))
    model.add(Convolution2D(64, 3, 3, activation=FLAGS.activation))
    model.add(Dropout(FLAGS.dropout))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def data_generator(image_paths, angles, batch_size=FLAGS.batch_size):
    while True:
        for offset in range(0, len(image_paths), batch_size):
            end = offset + batch_size
            yield helper.load_images(image_paths[offset:end]), angles[offset:end]


def train_model(model, train_generator, train_data_size, valid_generator, valid_data_size,
                callbacks, learning_rate=FLAGS.learning_rate, nb_epoch=FLAGS.nb_epoch):
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mse', 'acc'])
    return model.fit_generator(train_generator, samples_per_epoch=train_data_size,
                               validation_data=valid_generator, nb_val_samples=valid_data_size,
                               nb_epoch=nb_epoch, callbacks=callbacks, verbose=FLAGS.verbose)


def evaluate_model(model, x, y):
    loss, mean_squared_error, acc = model.evaluate(x, y, verbose=FLAGS.verbose)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def run(model, callbacks):
    image_paths, angles = helper.load_input_data(FLAGS.correction, FLAGS.angle_groups,
                                                 init_size=FLAGS.init_size,
                                                 init_shuffle=FLAGS.init_shuffle)

    # shuffle data by default
    train_image_paths, valid_image_paths, train_angles, valid_angles = train_test_split(image_paths,
                                                                                        angles,
                                                                                        test_size=FLAGS.validation_split,
                                                                                        random_state=0)

    train_data_size = len(train_image_paths) * 3
    valid_data_size = len(valid_image_paths) * 3

    train_generator = data_generator(train_image_paths, train_angles)
    valid_generator = data_generator(valid_image_paths, valid_angles)

    print("Train on {} samples".format(train_data_size))
    print("Validate on {} samples".format(valid_data_size))

    return train_model(model, train_generator, train_data_size, valid_generator, valid_data_size, callbacks)


def printFlags():
    print("Flags:")
    for key, value in FLAGS.__flags.items():
        print("{} = {}".format(key, value))


def init():
    helper.post_init()


def main(_):
    printFlags()
    init()

    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        verbose=0,
        mode='auto'
    )

    checkpoint = ModelCheckpoint(filepath=helper.MODEL_FILE, save_best_only=FLAGS.save_best_only,
                                 verbose=FLAGS.verbose)

    callbacks = [checkpoint, early_stopping]

    history = run(model, callbacks)

    # save history
    if history is not None:
        train_results = {}
        train_results.update(history.history)
        train_results["epoch"] = history.epoch
        helper.save_model_history(train_results)
        helper.save_args(FLAGS.__flags)

    print("Save state to archive")
    helper.archive_state(FLAGS.archive_name)

    print("Done")


if __name__ == "__main__":
    # ffmpeg -pattern_type glob -i "*.jpg" -vcodec mpeg4 -framerate 25 ../model-09-0.01.mp4
    # ffmpeg -pattern_type glob -i "*.jpg" -codec copy -framerate 25 ../model-09-0.01.avi
    tf.app.run()
