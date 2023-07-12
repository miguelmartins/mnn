import logging
import os

import h5py
import numpy as np
import tensorflow as tf

from data_processing.signal_extraction import DataExtractor, CircorExtractor
from data_processing.data_transformation import prepare_validation_data, get_train_test_indices, HybridPCGDataPreparer
from tqdm import tqdm
from custom_train_functions.hmm_train_step import hmm_train_step, hmm_train_step_nn_only
from loss_functions.mnn_losses import ForwardLoss
from models.custom_models import simple_convnet
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal
from utility_functions.parsing import get_fine_tune_parser

BASE_PATH = '../pretrained_models/circor_pretrain/'
logging.basicConfig(level=logging.INFO)

patch_size = 64
nch = 4

parser = get_fine_tune_parser(BASE_PATH)
args = parser.parse_args()

num_epochs = args.number_epochs
number_folders = args.number_folders
learning_rate = args.learning_rate
BASE_CKPT = args.model_directory

MODEL_CKTPT = os.path.join(BASE_CKPT, 'cnn_weights_fold_0')
TRANS_MAT_CKPT = os.path.join(BASE_CKPT, 'trans_mat_fold_0.npy')
P_STATES_CKPT = os.path.join(BASE_CKPT, 'p_states_fold_0.npy')

if args.hybrid:
    train_step_fn = hmm_train_step_nn_only
    logging.info('Using hybrid training.')
else:
    train_step_fn = hmm_train_step
    logging.info('Using static training.')

mnn_type = 'HYBRID' if args.hybrid else 'STATIC'
source, target = ('CIRCOR', 'PH16') if args.ph16 else ('PH16', 'CIRCOR')
NAME = f' FINE_TUNE_TO_{target}_MNN_{mnn_type}_{number_folders}fold_{num_epochs}ep_{learning_rate}lr'


def main():
    if args.ph16:
        good_indices, features, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets'
                                                                                                '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                           patch_size=patch_size)
    else:
        good_indices, features, labels, patient_ids = CircorExtractor.read_from_np(
            '../datasets/springer_circor_dataset.npy',
            patch_size=patch_size)

    experiment_logger = PCGExperimentLogger(path='../results/', name=NAME,
                                            number_folders=number_folders)

    h5_dir = os.path.join(experiment_logger.path, 'results.hdf5')
    h5_file = h5py.File(h5_dir, 'w')
    logging.info('Total number of valid sounds with length > ' +
                 str(patch_size / 50) +
                 ' seconds: ' + str(len(good_indices)))

    trans_mat = np.load(TRANS_MAT_CKPT)
    p_states = np.load(P_STATES_CKPT)
    model = simple_convnet(nch, patch_size)
    loss_object = ForwardLoss(tf.Variable(trans_mat, trainable=True, dtype=tf.float32),
                              tf.Variable(p_states, trainable=True, dtype=tf.float32))
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def load_configs():  # TODO: remove this nested function definition
        model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])
        model.load_weights(MODEL_CKTPT)  # Save initialization before training

        loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
        loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))

    for fold in range(number_folders):
        logging.info(f'Considering folder number: {fold + 1}')
        train_indices, test_indices = get_train_test_indices(good_indices=good_indices,
                                                             number_folders=number_folders,
                                                             patient_ids=patient_ids,
                                                             fold=fold)

        features_test = features[test_indices]
        labels_test = labels[test_indices]

        test_dp = HybridPCGDataPreparer(patch_size=patch_size,
                                        number_channels=nch,
                                        num_states=4)
        test_dp.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(test_dp,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      )
        labels_list, predictions_list = [], []

        # Viterbi algorithm in test set
        h5_file[f'/fold{fold}/data'] = f'{fold}'
        for i, (x, y) in tqdm(enumerate(test_dataset),
                              desc=f'fine tuning P(O) + viterbi decoding',
                              total=len(labels_test),
                              leave=True,
                              unit='sound',
                              postfix=f'{num_epochs} epochs per sound'):
            load_configs()
            y = y.numpy()
            raw_labels = np.argmax(y, axis=1).astype(np.int32)
            h5_file[f'/fold{fold}/data'].attrs[f'gt_{i}'] = raw_labels
            for it in range(num_epochs):
                _ = train_step_fn(model=model,
                                           optimizer=optimizer_nn,
                                           loss_object=loss_object,
                                           train_batch=x,
                                           label_batch=y,
                                           metrics=None)

                logits = model.predict(x)
                _, _, predictions = log_viterbi_no_marginal(loss_object.p_states.numpy(), loss_object.trans_mat.numpy(),
                                                            logits)
                h5_file[f'/fold{fold}/data'].attrs[f'preds{i}_{it}'] = predictions.astype(np.int32)

        # kept for compatability reasons with experiment_logger
        out_test = model.predict(test_dataset)
        length_sounds_test = np.array([len(ft) for ft in features_test])
        _, output_seqs = prepare_validation_data(out_test, test_indices, length_sounds_test)

        # collecting data and results
        experiment_logger.update_results(fold=fold,
                                         train_indices=train_indices,
                                         test_indices=test_indices,
                                         output_seqs=output_seqs,
                                         predictions=np.array(predictions_list, dtype=object),
                                         ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())
    h5_file.close()


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
