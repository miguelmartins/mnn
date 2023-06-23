import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from data_processing.signal_extraction import DataExtractor, CircorExtractor
from data_processing.data_transformation import HybridPCGDataPreparer2D, \
     get_train_test_indices
from tqdm import tqdm
from custom_train_functions.hmm_train_step import hmm_train_step, train_HMM_parameters, hmm_train_step_nn_only
from loss_functions.MMI_losses import CompleteLikelihoodLoss
from models.custom_models import simple_convnet2d
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal, QR_steady_state_distribution


def main():
    patch_size = 64
    nch = 76
    num_epochs = 50
    number_folders = 10
    learning_rate = 1e-3

    good_indices, patient_ids, features, labels = CircorExtractor.from_mat('../datasets/circor_final_labels50hz.mat',
                                                                           patch_size=patch_size)
    features = CircorExtractor.normalize_signal(features)
    features = DataExtractor.get_power_spectrum(data=features,
                                                sampling_rate=1000,
                                                window_length=150,
                                                window_overlap=130,
                                                window_type='hann')

    name = 'hmm_nnonly_cl_psd_circor'
    experiment_logger = PCGExperimentLogger(path='../results/rerun/hybrid/psd', name=name, number_folders=number_folders)
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))
    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints
    model = simple_convnet2d(nch, patch_size)
    loss_object = CompleteLikelihoodLoss(tf.Variable(tf.zeros((4, 4)), trainable=True, dtype=tf.float32),
                                         tf.Variable(tf.zeros((4,)), trainable=True, dtype=tf.float32))
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])
    model.save_weights('random_init')  # Save initialization before training

    acc_folds, prec_folds = [], []
    for j_fold in range(number_folders):
        min_val_loss = 1e100
        model.load_weights('random_init')  # Load random weights f.e. fold
        train_indices, test_indices = get_train_test_indices(good_indices=good_indices,
                                                             number_folders=number_folders,
                                                             patient_ids=patient_ids,
                                                             fold=j_fold)

        # remove from training data sounds that are from patient appearing in the testing set

        print('Considering folder number:', j_fold + 1)
        # This ia residual code from the initial implementation, kept for "panicky" reasons

        features_train = features[train_indices]
        features_test = features[test_indices]

        labels_train = labels[train_indices]
        labels_test = labels[test_indices]

        # This ia residual code from the initial implementation, kept for "panicky" reasons
        train_indices = good_indices[train_indices]
        test_indices = good_indices[test_indices]

        print('Number of training sounds:', len(labels_train))
        print('Number of testing sounds:', len(labels_test))

        X_train, X_dev, y_train, y_dev = train_test_split(
            features_train, labels_train, test_size=0.1, random_state=42)

        # _, trans_mat = train_HMM_parameters(y_train, one_hot=False)
       #  p_states = QR_steady_state_distribution(trans_mat)
        dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        dp.set_features_and_labels(X_train, y_train)
        train_dataset = tf.data.Dataset.from_generator(dp,
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                         dtype=tf.float32),
                                                           tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                       )
        dev_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        dev_dp.set_features_and_labels(X_dev, y_dev)
        dev_dataset = tf.data.Dataset.from_generator(dev_dp,
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                     )

        test_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        test_dp.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(test_dp,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      )

        # MLE Estimation for HMM
        dataset_np = list(train_dataset.as_numpy_iterator())
        dataset = np.array(dataset_np, dtype=object)
        labels_ = dataset[:, 1]
        p_states, trans_mat = train_HMM_parameters(labels_)
        loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
        loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))

        best_p_states = loss_object.p_states.numpy()
        best_trans_mat = loss_object.trans_mat.numpy()
        for ep in range(num_epochs):
            print('=', end='')
            metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)]
            for ep in range(num_epochs):
                print('=', end='')
                for i, (x_train, y_train) in tqdm(enumerate(train_dataset), desc=f'training', total=len(X_train),
                                                  leave=True):
                    hmm_train_step_nn_only(model=model,
                                   optimizer=optimizer_nn,
                                   loss_object=loss_object,
                                   train_batch=x_train,
                                   label_batch=y_train,
                                   metrics=metrics)


            # 29500        janela: 150ms    stride: 50ms (overlap=100ms)
            #              stride = 1 / (50hz) = 20 ms /(o intervalo tem que coincidir com o periodo)
            #              N_janelas = (|Tobs + pad| - janela) / stride + 1   -> (N_janelas , numero_freq)
            #     sinal 10 valores -> psd com stride 2 -> (0, 2, 4, 6, 8); janela = 5 (0, 2, 4, 6
            #     PSD deve ter output igual as anotacoes, dado que usamos o stride correcto!
            (train_loss, train_acc) = model.evaluate(train_dataset, verbose=0)
            print("Train Loss", train_loss, "Train_acc", train_acc)
            (val_loss, val_acc) = model.evaluate(dev_dataset, verbose=0)
            print("Epoch ", ep, " train loss = ", train_loss, " train accuracy = ", train_acc, "val loss = ", val_loss,
                  "val accuracy = ", val_acc)
            checkpoint = './checkpoints/hybrid_physion/' + str(j_fold) + '/my_checkpoint'
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

        labels_list, cnn_list, predictions_list = [], [], []
        print(loss_object.p_states.numpy())
        print(loss_object.trans_mat.numpy())
        acc_cnn = []
        # Viterbi algorithm in test set
        for x, y in tqdm(test_dataset, desc=f'validating (viterbi)', total=len(labels_test), leave=True):
            logits = model.predict(x)
            y = y.numpy()
            _, _, predictions = log_viterbi_no_marginal(loss_object.p_states.numpy(), loss_object.trans_mat.numpy(),
                                                        logits)
            predictions = predictions.astype(np.int32)
            cnn_preds = np.argmax(logits, axis=1)
            cnn_list.append(cnn_preds)
            raw_labels = np.argmax(y, axis=1).astype(np.int32)
            predictions_list.append(predictions)
            labels_list.append(raw_labels)
            acc = accuracy_score(raw_labels, predictions)
            acc_cnn.append(accuracy_score(raw_labels, cnn_preds))
            prc = precision_score(raw_labels, predictions, average=None)
            accuracy.append(acc)
            precision.append(prc)
        print("CNN-> Average Accuracy", np.mean(acc_cnn))
        print("Viterbi-> Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
        acc_folds.append(np.mean(acc))
        prec_folds.append(np.mean(prc))

        # collecting data and results
        experiment_logger.update_results(fold=j_fold,
                                         train_indices=train_indices,
                                         test_indices=test_indices,
                                         output_seqs=np.array(cnn_list, dtype=object),
                                         predictions=np.array(predictions_list, dtype=object),
                                         ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()

