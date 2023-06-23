import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_processing.signal_extraction import DataExtractor, CircorExtractor
from data_processing.data_transformation import HybridPCGDataPreparer, prepare_validation_data, get_train_test_indices
from custom_train_functions.hmm_train_step import hmm_train_step, train_HMM_parameters, hmm_train_step_multi_opt, \
    hmm_train_step_nn_only
from loss_functions.MMI_losses import MMILoss, CompleteLikelihoodLoss
from models.custom_models import simple_convnet
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal, QR_steady_state_distribution


def main():

    patch_size = 64
    nch = 4
    num_epochs = 50
    number_folders = 10
    learning_rate = 1e-3

    # Read Circor #
    good_indices, features, labels, patient_ids = CircorExtractor.read_from_np(
        '../datasets/PCG/circor_final/springer_circor_dataset.npy',
        patch_size=patch_size)
    ######

    # Read Ph16
    good_indices_ph16, features_ph16, labels_ph16, patient_ids_ph16, length_sounds_ph16 = DataExtractor.extract(
        path='../datasets/PCG/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
        patch_size=patch_size)

    name = "merged_completelikelihood_nnonly"
    experiment_logger = PCGExperimentLogger(path='../results/merged', name=name, number_folders=number_folders)

    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints
    model = simple_convnet(nch, patch_size)
    loss_object = CompleteLikelihoodLoss(tf.Variable(tf.zeros((4, 4)), trainable=True, dtype=tf.float32),
                                         tf.Variable(tf.zeros((4,)), trainable=True, dtype=tf.float32))
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])
    model.save_weights('random_init')  # Save initialization before training

    acc_folds, prec_folds = [], []
    for j_fold in range(number_folders):
        min_val_loss = 1e100
        model.load_weights('random_init')  # Load random weights f.e. fold
        train_indices, test_indices = get_train_test_indices(good_indices=good_indices, number_folders=number_folders,
                                                             patient_ids=patient_ids, fold=j_fold)
        train_indices_ph16, test_indices_ph16 = get_train_test_indices(good_indices=good_indices_ph16,
                                                                       number_folders=number_folders,
                                                                       patient_ids=patient_ids_ph16, fold=j_fold)

        # remove from training data sounds that are from patient appearing in the testing set

        print('Considering folder number:', j_fold + 1)

        def process_indices(features, labels, good_indices, train_indices, test_indices):
            features_train = features[train_indices]
            features_test = features[test_indices]

            labels_train = labels[train_indices]
            labels_test = labels[test_indices]

            # This ia residual code from the initial implementation, kept for "panicky" reasons
            train_indices = good_indices[train_indices]
            test_indices = good_indices[test_indices]
            return features_train, features_test, labels_train, labels_test, train_indices, test_indices

        ft_train_ph16, ft_test_ph16, l_train_ph16, l_test_ph16, train_indices_ph16, test_indices_ph16 = process_indices(
            features_ph16,
            labels_ph16,
            good_indices_ph16,
            train_indices_ph16,
            test_indices_ph16)

        ft_train_circor, ft_test_circor, l_train_circor, l_test_circor, train_indices_circor, test_indices_circor = process_indices(
            features,
            labels,
            good_indices,
            train_indices,
            test_indices)

        features_train = np.concatenate([ft_train_ph16, ft_train_circor])
        features_test = np.concatenate([ft_test_ph16, ft_test_circor])

        labels_train = np.concatenate([l_train_ph16, l_train_circor])
        labels_test = np.concatenate([l_test_ph16, l_test_circor])

        train_indices = np.concatenate([train_indices_ph16, train_indices_circor])
        test_indices = np.concatenate([test_indices_ph16, test_indices_circor])

        print('Number of training sounds:', len(labels_train))
        print('Number of testing sounds:', len(labels_test))

        X_train, X_dev, y_train, y_dev = train_test_split(
            features_train, labels_train, test_size=0.1, random_state=42)

        dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
        dp.set_features_and_labels(X_train, y_train)
        train_dataset = tf.data.Dataset.from_generator(dp,
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None, patch_size, nch),
                                                                         dtype=tf.float32),
                                                           tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                       ).prefetch(buffer_size=tf.data.AUTOTUNE)
        dev_dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
        dev_dp.set_features_and_labels(X_dev, y_dev)
        dev_dataset = tf.data.Dataset.from_generator(dev_dp,
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(None, patch_size, nch), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                     ).prefetch(buffer_size=tf.data.AUTOTUNE)

        test_dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
        test_dp.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(test_dp,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      ).prefetch(buffer_size=tf.data.AUTOTUNE)

        # MLE Estimation for HMM
        dataset_np = list(train_dataset.as_numpy_iterator())
        dataset = np.array(dataset_np, dtype=object)
        labels_ = dataset[:, 1]
        _, trans_mat = train_HMM_parameters(labels_)
        p_states = QR_steady_state_distribution(trans_mat)
        loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
        loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))

        train_dataset = train_dataset.shuffle(len(X_train), reshuffle_each_iteration=True)

        best_p_states = loss_object.p_states.numpy()
        best_trans_mat = loss_object.trans_mat.numpy()
        metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)]
        print(loss_object.trans_mat.numpy())
        for ep in range(num_epochs):
            print('=', end='')
            metrics[0].reset_states()
            for i, (x_train, y_train) in tqdm(enumerate(train_dataset), desc=f'training', total=len(X_train),
                                              leave=True):
                hmm_train_step_nn_only(model=model,
                                       optimizer=optimizer_nn,
                                       loss_object=loss_object,
                                       train_batch=x_train,
                                       label_batch=y_train,
                                       metrics=metrics)

            print(f"ep: {ep}. Cat accuracy: {metrics[0].result().numpy()}")
            print(loss_object.trans_mat.numpy())
            print(loss_object.p_states.numpy())
            # check performance at each epoch
            (train_loss, train_acc) = model.evaluate(train_dataset, verbose=0)
            print("Train Loss", train_loss, "Train_acc", train_acc)
            (val_loss, val_acc) = model.evaluate(dev_dataset, verbose=0)
            print("Epoch ", ep, " train loss = ", train_loss, " train accuracy = ", train_acc, "val loss = ", val_loss,
                  "val accuracy = ", val_acc)
            checkpoint = './checkpoints/hybrid_physion/' + str(j_fold) + '/my_checkpoint'
            print(loss_object.trans_mat.numpy())
            print(loss_object.p_states.numpy())
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # SAVE MODEL
                best_p_states = loss_object.p_states.numpy()
                best_trans_mat = loss_object.trans_mat.numpy()
                experiment_logger.save_model_checkpoints(model, best_p_states, best_trans_mat, '/cnn_weights_fold_',
                                                         j_fold)

        model = experiment_logger.load_model_checkpoint_weights(model)
        loss_object.p_states.assign(tf.Variable(best_p_states, trainable=True, dtype=tf.float32))
        loss_object.trans_mat.assign(tf.Variable(best_trans_mat, trainable=True, dtype=tf.float32))
        out_test = model.predict(test_dataset)
        accuracy, precision = [], []

        labels_list, predictions_list = [], []
        print(loss_object.p_states.numpy())
        print(loss_object.trans_mat.numpy())

        # Viterbi algorithm in test set
        for x, y in tqdm(test_dataset, desc=f'validating (viterbi)', total=len(labels_test), leave=True):
            logits = model.predict(x)
            y = y.numpy()
            _, _, predictions = log_viterbi_no_marginal(loss_object.p_states.numpy(), loss_object.trans_mat.numpy(),
                                                        logits)
            predictions = predictions.astype(np.int32)
            raw_labels = np.argmax(y, axis=1).astype(np.int32)
            predictions_list.append(predictions)
            labels_list.append(raw_labels)
            acc = accuracy_score(raw_labels, predictions)
            prc = precision_score(raw_labels, predictions, average=None)
            accuracy.append(acc)
            precision.append(prc)
        print("Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
        acc_folds.append(np.mean(acc))
        prec_folds.append(np.mean(prc))
        length_sounds_test = np.zeros(len(features_test))
        for j in range(len(features_test)):
            length_sounds_test[j] = len(features_test[j])

        # recover sound labels from patch labels
        output_probs, output_seqs = prepare_validation_data(out_test, test_indices, length_sounds_test)

        sample_acc = np.zeros((len(labels_test),))
        for j in range(len(labels_test)):
            sample_acc[j] = 1 - (np.sum((output_seqs[j] != labels_test[j] - 1).astype(int)) / len(labels_test[j]))

        print('Test mean sample accuracy for this folder:', np.sum(sample_acc) / len(sample_acc))
        for j in range(len(labels_test)):
            sample_acc[j] = 1 - (
                    np.sum((predictions_list[j] != labels_test[j] - 1).astype(int)) / len(labels_test[j]))
        print("Viterbi: ", np.sum(sample_acc) / len(sample_acc))

        # collecting data and results
        experiment_logger.update_results(fold=j_fold,
                                         train_indices=train_indices,
                                         test_indices=test_indices,
                                         output_seqs=output_seqs,
                                         predictions=np.array(predictions_list, dtype=object),
                                         ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
