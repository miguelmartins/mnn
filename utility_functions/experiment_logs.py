import datetime
import pickle

import numpy as np
from pathlib import Path
import scipy.io as sio


class PCGExperimentLogger:
    def __init__(self, path, name, number_folders):
        self.train_indices_cell = np.ndarray(shape=(number_folders,), dtype=np.ndarray)
        self.test_indices_cell = np.ndarray(shape=(number_folders,), dtype=np.ndarray)
        self.out_ground_truth_cell = np.ndarray(shape=(number_folders,), dtype=np.ndarray)
        self.out_vit_seq_cell = np.ndarray(shape=(number_folders,), dtype=np.ndarray)
        self.out_seq_cell = np.ndarray(shape=(number_folders,), dtype=np.ndarray)
        date_exp_ = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        name_ = path + '/' + name + '/' + date_exp_
        save_dir = Path(name_)
        save_dir.mkdir(parents=True)
        self.path = name_
        self.last_checkpoint = None

    @staticmethod
    def stored_model_checkpoints(model, path, cnn_prefix, fold):
        path_ = path + '/'
        cnn_path = path_ + cnn_prefix + str(fold)
        model.load_weights(cnn_path)
        return model, np.load(path_ + 'p_states_fold_' + str(fold) + '.npy'), np.load(
            path_ + 'trans_mat_fold_' + str(fold) + '.npy')

    def save_model_checkpoints(self, model, p_states, trans_mat, checkpoint_path, fold):
        self.last_checkpoint = self.path + '/' + checkpoint_path + str(fold)
        model.save_weights(self.last_checkpoint)
        _path = self.path + '/'
        np.save(_path + 'p_states_fold_' + str(fold), p_states)
        np.save(_path + 'trans_mat_fold_' + str(fold), trans_mat)

    def save_markov_state(self, fold, p_states, trans_mat):
        _path = self.path + '/'
        np.save(_path + 'p_states_fold_' + str(fold), p_states)
        np.save(_path + 'trans_mat_fold_' + str(fold), trans_mat)

    def load_model_checkpoint_weights(self, model):
        model.load_weights(self.last_checkpoint)
        return model

    def update_results(self, *, fold, train_indices, test_indices, output_seqs, predictions, ground_truth):
        self.train_indices_cell[fold] = train_indices
        self.test_indices_cell[fold] = test_indices
        self.out_seq_cell[fold] = output_seqs
        self.out_vit_seq_cell[fold] = predictions
        self.out_ground_truth_cell[fold] = ground_truth

    def save_results(self, *, p_states, trans_mat):
        path_ = self.path + '/'
        sio.savemat(path_ + 'train_indexes.mat',
                    {'train_indexes': self.train_indices_cell})
        sio.savemat(path_ + 'test_indexes.mat', {'test_indexes': self.test_indices_cell})
        sio.savemat(path_ + 'p_states.mat', {'p_states': p_states})
        sio.savemat(path_ + 'trans_mat.mat', {'trans_mat': trans_mat})
        sio.savemat(path_ + 'out_seq.mat', {'out_seq': self.out_seq_cell})
        sio.savemat(path_ + 'ground_truth.mat',
                    {'ground_truth': self.out_ground_truth_cell})
        sio.savemat(path_ + 'viterbi.mat', {'viterbi': self.out_vit_seq_cell})


def checkpoint_model_at_fold(*, val_loss, min_val_loss, best_p_states, best_trans_mat, experiment_logger, loss_object, model, fold):
    val_loss_ = min_val_loss
    if val_loss < min_val_loss:
        val_loss_ = val_loss
        # SAVE MODEL
        best_p_states = loss_object.p_states.numpy()
        best_trans_mat = loss_object.trans_mat.numpy()
        experiment_logger.save_model_checkpoints(model, best_p_states, best_trans_mat, '/cnn_weights_fold_',
                                                 fold)
    return val_loss_, best_p_states, best_trans_mat
