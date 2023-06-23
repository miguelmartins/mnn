import numpy as np
import tensorflow as tf


class PCGDataPreparer:
    def __init__(self, patch_size, stride, number_channels, num_states=4):
        self.patch_size = patch_size
        self.stride = stride
        self.number_channels = number_channels
        self.num_states = num_states
        self.features = None
        self.labels = None

    def _split_PCG_features(self, sound, label):
        num_samples = len(sound)
        num_windows = int((num_samples - self.patch_size) / self.stride) + 1
        _sound = np.zeros((num_windows, self.patch_size, self.number_channels))
        _labels = np.zeros((num_windows, self.patch_size, self.num_states))
        for window_idx in range(num_windows):
            patch_start = window_idx * self.stride
            _sound[window_idx, :] = sound[patch_start:patch_start + self.patch_size, :]
            _labels[window_idx, :] = label[patch_start: patch_start + self.patch_size, :]

        window_remain = num_samples - self.patch_size
        if window_remain % self.stride > 0:
            _sound = np.concatenate((_sound, [sound[window_remain:num_samples, :]]), axis=0)
            _labels = np.concatenate((_labels, [label[window_remain:num_samples, :]]), axis=0)

        return _sound, _labels

    def set_features_and_labels(self, features, labels):
        self.features = features
        self.labels = labels

    def __call__(self):
        num_observations = len(self.labels)
        for obs_idx in range(num_observations):
            sound_ = self.features[obs_idx]
            label = self.labels[obs_idx] - 1
            one_hot_label = np.zeros((len(label), self.num_states))
            for state in range(self.num_states):
                one_hot_label[:, state] = (label == state).astype(int)

            sound, labels_window = self._split_PCG_features(sound_, one_hot_label)
            for s, l in zip(sound, labels_window):
                yield s.astype('float32'), l.astype('float32')


class IndexedPCGDataPreparer(PCGDataPreparer):
    def __init__(self, patch_size, stride, number_channels, num_states=4):
        super().__init__(patch_size, stride, number_channels, num_states)

    def __call__(self):
        num_observations = len(self.labels)
        for obs_idx in range(num_observations):
            sound_ = self.features[obs_idx]
            label = self.labels[obs_idx] - 1
            one_hot_label = np.zeros((len(label), self.num_states))
            for state in range(self.num_states):
                one_hot_label[:, state] = (label == state).astype(int)

            sound, _ = self._split_PCG_features(sound_, one_hot_label)
            yield sound, one_hot_label

    def get_averaged_prediction(self, target, y_pred):
        length = int(round(len(target)))
        number_patches = int((length - self.patch_size) / self.stride) + 1
        if (length - self.patch_size) % self.stride > 0:
            number_patches = int(round(number_patches + 1))

        p_index = 0
        logits = np.zeros((number_patches, length, self.num_states))
        for i in range(int((length - self.patch_size) / self.stride) + 1):
            logits[i, i * self.stride:i * self.stride + self.patch_size, :] = y_pred[p_index, :, :]
            p_index += 1

        if (length - self.patch_size) % self.stride > 0:
            logits[number_patches - 1, length - self.patch_size:length, :] = y_pred[p_index, :, :]
            p_index += 1

        logits = np.sum(logits, axis=0)

        prob_sum = np.sum(logits, axis=1)
        prob_sum = np.tile(prob_sum, (4, 1))
        prob_sum = np.transpose(prob_sum)

        logits = np.divide(logits, prob_sum)
        predictions = np.argmax(logits, axis=1)

        return logits, predictions


class HybridPCGDataPreparer(PCGDataPreparer):
    def __init__(self, patch_size, number_channels, num_states=4):
        super().__init__(patch_size, 0, number_channels, num_states)

    def _split_PCG_features(self, sound, label):
        n = len(sound)
        x_padded_temp = np.concatenate((np.zeros((int(self.patch_size / 2), self.number_channels)), sound), axis=0)
        x_padded = np.concatenate((x_padded_temp, np.zeros((int(self.patch_size / 2) - 1, self.number_channels))),
                                  axis=0)
        x_array = np.zeros((n, self.patch_size, self.number_channels))
        s_array = np.zeros((n, 4))
        for j in range(0, n):
            x_array[j, :, :] = x_padded[j:j + self.patch_size, :]
            s_array[j, :] = label[j, :]
        return x_array, s_array

    def __call__(self):
        num_observations = len(self.labels)
        for obs_idx in range(num_observations):
            sound_ = self.features[obs_idx]
            label = self.labels[obs_idx] - 1
            one_hot_label = np.zeros((len(label), self.num_states))
            for state in range(self.num_states):
                one_hot_label[:, state] = (label == state).astype(int)

            sound, labels = self._split_PCG_features(sound_, one_hot_label)
            yield sound, labels


class HybridPCGDataPreparer2D(HybridPCGDataPreparer):
    def __init__(self, patch_size, number_channels, num_states=4):
        super().__init__(patch_size, number_channels, num_states)

    def _split_PCG_features(self, sound, label):
        n = len(label)
        sound = sound.reshape(sound.shape + (1,))
        x_padded_temp = np.concatenate((np.zeros((int(self.patch_size / 2), self.number_channels, 1)), sound), axis=0)
        x_padded = np.concatenate((x_padded_temp, np.zeros((int(self.patch_size / 2) - 1, self.number_channels, 1))),
                                  axis=0)
        x_array = np.zeros((n, self.patch_size, self.number_channels, 1))
        s_array = np.zeros((n, 4))
        for j in range(0, n):
            x_array[j, :, :] = x_padded[j:j + self.patch_size, :]
            s_array[j, :] = label[j, :]
        return x_array, s_array


def get_data_from_generator(*, data_processor, batch_size, patch_size, number_channels, number_classes, trainable=True):
    data = tf.data.Dataset.from_generator(data_processor,
                                          output_signature=(
                                              tf.TensorSpec(shape=(patch_size, number_channels), dtype=tf.float32),
                                              tf.TensorSpec(shape=(patch_size, number_classes), dtype=tf.float32))
                                          )
    if trainable:
        data = data.shuffle(5000, reshuffle_each_iteration=True)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def get_data_from_hybrid_generator(*, data_processor, batch_size, patch_size, number_channels, number_classes,
                                   trainable=True):
    data = tf.data.Dataset.from_generator(data_processor,
                                          output_signature=(
                                              tf.TensorSpec(shape=(None, patch_size, number_channels, 1),
                                                            dtype=tf.float32),
                                              tf.TensorSpec(shape=(None, number_classes), dtype=tf.float32))
                                          )
    if trainable:
        data = data.shuffle(5000, reshuffle_each_iteration=True)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def unet_prepare_validation_data(out_test, test_indexes, length_sounds, patch_size, stride):
    output_probs = np.ndarray(shape=(len(test_indexes),), dtype=np.ndarray)
    output_seqs = np.ndarray(shape=(len(test_indexes),), dtype=np.ndarray)

    p_index = 0

    for j in range(len(test_indexes)):
        T = int(round(length_sounds[test_indexes[j]]))
        # number of patches associated to this sound
        N_patches = int((T - patch_size) / stride) + 1
        if (T - patch_size) % stride > 0:
            N_patches = int(round(N_patches + 1))

        prob_out = np.zeros((N_patches, T, 4))

        for i in range(int((T - patch_size) / stride) + 1):
            prob_out[i, i * stride:i * stride + patch_size, :] = out_test[p_index, :, :]
            p_index += 1

        if (T - patch_size) % stride > 0:
            prob_out[N_patches - 1, T - patch_size:T, :] = out_test[p_index, :, :]
            p_index += 1

        prob_out = np.sum(prob_out, axis=0)

        prob_sum = np.sum(prob_out, axis=1)
        prob_sum = np.tile(prob_sum, (4, 1))
        prob_sum = np.transpose(prob_sum)

        prob_out = np.divide(prob_out, prob_sum)
        seq_out = np.argmax(prob_out, axis=1)

        output_probs[j] = prob_out
        output_seqs[j] = seq_out

    return output_probs, output_seqs


def prepare_validation_data(out_test, test_indexes, length_sounds_test):
    output_probs = np.ndarray(shape=(len(test_indexes),), dtype=np.ndarray)
    output_seqs = np.ndarray(shape=(len(test_indexes),), dtype=np.ndarray)

    p_index = 0

    for j in range(len(test_indexes)):
        T = int(round(length_sounds_test[j]))  # length of a given sound

        prob_out = out_test[p_index:p_index + T, :]
        p_index += T

        seq_out = np.argmax(prob_out, axis=1)

        output_probs[j] = prob_out
        output_seqs[j] = seq_out

    return output_probs, output_seqs


def get_train_test_indices(*, good_indices, number_folders, patient_ids, fold):
    fold_dim = int(len(good_indices) / number_folders)

    total_indices = np.array(range(number_folders * fold_dim))
    test_indices = np.array(range(fold * fold_dim, (fold + 1) * fold_dim))
    train_indices = np.delete(total_indices, test_indices)

    # remove from training data sounds that are from patient appearing in the test set
    train_indexes_norep = []
    for idx in range(len(train_indices)):
        if not (patient_ids[train_indices[idx]] in patient_ids[test_indices]):
            train_indexes_norep.append(train_indices[idx])

    train_indices = np.array(train_indexes_norep)

    return train_indices, test_indices


def get_fold_indices(dataset, n=10):
    #  Calculate the splits in a 1D array
    fold_indices = np.arange(start=0, step=int(np.floor(dataset.shape[0] / n)), stop=dataset.shape[0])
    if fold_indices[-1] != len(dataset):  # Corner case when fold dim is not divisible by length of dataset
        fold_indices[-1] = len(dataset)
    all_indices = np.arange(len(dataset))

    # Use the fold splits in 2x1 groups to set the indices range for test
    test_idx = np.array(
        [np.arange(start=fold_indices[i], stop=fold_indices[i + 1]) for i in range(len(fold_indices) - 1)],
        dtype=object)

    train_idx = []  # Remove by set operation ALL_INDICES - TEST_INDICES f.e. fold
    for i in range(n):
        train_idx.append(np.delete(all_indices, test_idx[i]))
    train_idx = np.array(train_idx, dtype=object)
    return train_idx, test_idx
