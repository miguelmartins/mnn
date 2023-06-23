import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from data_processing.signal_extraction import DataExtractor
from data_processing.data_transformation import HybridPCGDataPreparer2D, \
    prepare_validation_data, get_train_test_indices
from tqdm import tqdm
from custom_train_functions.hmm_train_step import hmm_train_step, train_HMM_parameters
from loss_functions.MMI_losses import CompleteLikelihoodLoss
from models.custom_models import simple_convnet2d
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal


def main():
    patch_size = 64
    nch = 76
    num_epochs = 1
    learning_rate = 1e-3
    split = .1
    number_folders = 1

    good_indices, _, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets/PCG'
                                                                                     '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                patch_size=patch_size)
    features_, _ = DataExtractor.read_physionet_mat(
        '../datasets/PCG/example_data.mat')  # TODO: o som original deve ter 20x mais samples
    length_sounds = np.array([len(features_[j]) for j in range(len(features_))])
    features = DataExtractor.get_power_spectrum(data=features_,
                                                sampling_rate=1000,
                                                window_length=150,
                                                window_overlap=130,
                                                window_type='hann')
    features = DataExtractor.filter_by_index(features, good_indices)
    print("Debug", len(features), len(labels), len(good_indices))
    name = 'hmm_completlikelihood1e3_physio16_psd_joint'
    experiment_logger = PCGExperimentLogger(path='../results/hybrid/ph_holdout', name=name,
                                            number_folders=number_folders)
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

    min_val_loss = 1e3
    model.load_weights('random_init')  # Load random weights f.e. fold
    # remove from training data sounds that are from patient appearing in the testing set
    # TODO: make this right, perhaps at start
    # features = features[good_indices]
    # labels = labels[good_indices]
    print("lengths", len(features), len(labels), len(patient_ids))
    indices = np.arange(len(features))
    idx_train, idx_test = train_test_split(indices, test_size=split, random_state=42)
    print("split", len(idx_train), len(idx_test))
    idx_train_ = []
    for idx in idx_train:
        if patient_ids[idx] not in patient_ids[idx_test]:
            idx_train_.append(idx)
    idx_train = np.array(idx_train_)
    X_train, y_train = features[idx_train], labels[idx_train]
    X_dev, y_dev = features[idx_test], labels[idx_test]

    # NORMALIZAR PSD
    # como separar as features para a nossa CNN?
    # com os envolopes separámos em patches, aqui usamos a própria dimensão da STFT?
    # Contruir datapreparer:
    #    - Separe o som por janelas PSD
    #
    # Implementar uma CNN com convoluções 2D: first approach -> mudar a nossa CNN para operações 2D.
    # ???
    # Profit
    dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
    dp.set_features_and_labels(X_train, y_train)
    train_dataset = tf.data.Dataset.from_generator(dp,
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                     dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                   ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    dev_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
    dev_dp.set_features_and_labels(X_dev, y_dev)
    dev_dataset = tf.data.Dataset.from_generator(dev_dp,
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(None, patch_size, nch, 1), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                 ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # MLE Estimation for HMM
    dataset_np = list(train_dataset.as_numpy_iterator())
    dataset = np.array(dataset_np, dtype=object)
    labels_ = dataset[:, 1]
    p_states, trans_mat = train_HMM_parameters(labels_)
    loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
    loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))

    for ep in range(num_epochs):
        print('=', end='')
        for i, (x_train, y_train) in tqdm(enumerate(train_dataset), desc=f'training', total=len(X_train), leave=True):
            hmm_train_step(model=model,
                           optimizer=optimizer_nn,
                           loss_object=loss_object,
                           train_batch=x_train,
                           label_batch=y_train,
                           metrics=None)

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
        checkpoint = './checkpoints/hybrid_physion/ph_holdout/my_checkpoint'
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            # SAVE MODEL
            best_p_states = loss_object.p_states.numpy()
            best_trans_mat = loss_object.trans_mat.numpy()
            experiment_logger.save_model_checkpoints(model, best_p_states, best_trans_mat, '/cnn_weights_fold_',
                                                     0)

    model = experiment_logger.load_model_checkpoint_weights(model)
    loss_object.p_states.assign(tf.Variable(best_p_states, trainable=True, dtype=tf.float32))
    loss_object.trans_mat.assign(tf.Variable(best_trans_mat, trainable=True, dtype=tf.float32))
    out_test = model.predict(dev_dataset)
    accuracy, precision = [], []

    labels_list, cnn_list, predictions_list = [], [], []
    print(loss_object.p_states.numpy())
    print(loss_object.trans_mat.numpy())
    acc_cnn = []
    # Viterbi algorithm in test set
    for x, y in tqdm(dev_dataset, desc=f'validating (viterbi)', total=len(y_dev), leave=True):
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
    # TODO: get training indices

    experiment_logger.update_results(fold=0,
                                     train_indices=good_indices,
                                     test_indices=idx_test,
                                     output_seqs=np.array(cnn_list, dtype=object),
                                     predictions=np.array(predictions_list, dtype=object),
                                     ground_truth=np.array(labels_list, dtype=object))
    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())


if __name__ == '__main__':
    main()
