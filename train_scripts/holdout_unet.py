import tensorflow as tf
import numpy as np
import os
import datetime

from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from custom_train_functions.hmm_train_step import train_HMM_parameters
from data_processing.signal_extraction import DataExtractor
from models.custom_models import unet_pcg
from data_processing.data_transformation import PCGDataPreparer, HybridPCGDataPreparer, \
    unet_prepare_validation_data, get_data_from_generator, get_train_test_indices
from utility_functions.experiment_logs import PCGExperimentLogger
from utility_functions.hmm_utilities import log_viterbi_no_marginal
from utility_functions.metrics import get_metrics


def main():
    patch_size = 64
    stride = 8
    nch = 4
    number_folders = 1
    learning_rate = 1e-4
    split = .2
    EPOCHS = 50
    BATCH_SIZE = 1

    dp = PCGDataPreparer(patch_size=patch_size, stride=stride, number_channels=nch, num_states=4)
    dp_dev = PCGDataPreparer(patch_size=patch_size, stride=stride, number_channels=nch, num_states=4)

    good_indices, features, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets/PCG'
                                                                                            '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                       patch_size=patch_size)
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))

    experiment_logger = PCGExperimentLogger(path='../results/rerun/unet/ph_holdout', name='unet',
                                            number_folders=number_folders)

    model = unet_pcg(nch, patch_size)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.save_weights('random_init_unet')
    savedir = 'tmpCC_' + datetime.datetime.now().strftime("%Y_%m_%d")

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    acc_folds, prec_folds = [], []
    model.load_weights('random_init_unet')
    indices = np.arange(len(features))
    idx_train, idx_test = train_test_split(indices, test_size=split, random_state=42)
    idx_train_ = []
    for idx in range(len(idx_train)):
        if not (patient_ids[idx_train[idx]] in patient_ids[idx_test]):
           idx_train_.append(idx_train[idx])
    idx_train = np.array(idx_train_)
    X_train, y_train = features[idx_train], labels[idx_train]
    X_dev, y_dev = features[idx_test], labels[idx_test]

    dp.set_features_and_labels(X_train, y_train)

    train_dataset = get_data_from_generator(data_processor=dp,
                                            batch_size=BATCH_SIZE,
                                            patch_size=patch_size,
                                            number_channels=nch,
                                            number_classes=4,
                                            trainable=False)  # I am not sure about the shuffling, might remove it

    dp_dev.set_features_and_labels(X_dev, y_dev)
    dev_dataset = get_data_from_generator(data_processor=dp_dev,
                                          batch_size=BATCH_SIZE,
                                          patch_size=patch_size,
                                          number_channels=nch,
                                          number_classes=4,
                                          trainable=False)

    # Use the HMM DataPreparer to extract the labels for the HMM MLE
    hmm_dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
    hmm_dp.set_features_and_labels(X_train, y_train)
    train_dataset_ = tf.data.Dataset.from_generator(hmm_dp,
                                                    output_signature=(
                                                        tf.TensorSpec(shape=(None, patch_size, nch),
                                                                      dtype=tf.float32),
                                                        tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                    )

    dataset_np = list(train_dataset_.as_numpy_iterator())
    dataset = np.array(dataset_np, dtype=object)
    labels_ = dataset[:, 1]
    print(y_train.shape, labels_.shape)

    p_states, trans_mat = train_HMM_parameters(labels_)

    checkpoint_path = experiment_logger.path + '/weights_.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    history = model.fit(train_dataset, validation_data=dev_dataset, validation_steps=1, epochs=EPOCHS, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint])
    experiment_logger.save_markov_state(0, p_states, trans_mat)
    model.load_weights(checkpoint_path)
    # prediction on test data

    hmm_dp_test = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
    hmm_dp_test.set_features_and_labels(X_dev, y_dev)
    test_dataset = tf.data.Dataset.from_generator(hmm_dp_test,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, patch_size, nch),
                                                                    dtype=tf.float32),
                                                      tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                  )

    out_test = model.predict(dev_dataset)
    # recover sound labels from patch labels
    output_probs, output_seqs = unet_prepare_validation_data(out_test, idx_test, length_sounds, patch_size,
                                                             stride)
    accuracy, precision = [], []
    i = 0
    labels_list, predictions_list = [], []

    for x, y in test_dataset:
        logits = output_probs[i]
        y = y.numpy()
        _, _, predictions = log_viterbi_no_marginal(p_states, trans_mat, logits)
        predictions = predictions.astype(np.int32)
        labels_ = np.argmax(y, axis=1).astype(np.int32)
        predictions_list.append(predictions)
        labels_list.append(labels_)
        acc = accuracy_score(labels_, predictions)
        prc = precision_score(labels_, predictions, average=None)
        accuracy.append(acc)
        precision.append(prc)
        i += 1
    print("Testing shape", output_seqs.shape, "number of x in test_dataset", i, "labels shape", len(labels_list))
    print("Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
    acc_folds.append(np.mean(accuracy))
    prec_folds.append(np.mean(precision))
    sample_acc = np.zeros((len(y_dev),))
    for j in range(len(y_dev)):
        sample_acc[j] = 1 - (np.sum((output_seqs[j] != y_dev[j] - 1).astype(int)) / len(y_dev[j]))

    print('Sounds mean sample accuracy for this folder:', np.sum(sample_acc) / len(sample_acc))
    ppv, sens, acc = get_metrics(labels_list, output_seqs)

    # collecting data and results
    experiment_logger.update_results(fold=0,
                                     train_indices=good_indices,
                                     test_indices=idx_test,
                                     output_seqs=output_seqs,
                                     predictions=np.array(predictions_list, dtype=object),
                                     ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=p_states, trans_mat=trans_mat)


if __name__ == "__main__":
    with tf.device('/cpu:0'):
        main()
