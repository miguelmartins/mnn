import random
import numpy as np


def synthetic_HMMGMM_vectors(trans_mat, m_emis, s_emis, *, sigma_noise=1, T=100, nch=64):
    states = np.empty((0))
    observations = np.empty((0, nch))

    # random.seed(SEED)
    state = int(random.randint(0, 3))  # state 1

    for aux_ in range(T):
        # temp_s=(state)*np.ones((int(max(1,np.random.normal(m_sojourn[state],s_sojourn[state])))))
        states = np.concatenate((states, np.array([state])), axis=0)
        # np.random.seed(SEED * aux_)
        new_observation = np.random.normal(loc=m_emis[int(state)], scale=s_emis[int(state)], size=(1, nch))
        observations = np.concatenate((observations, new_observation), axis=0)

        # random.seed(SEED)
        coin = random.random()  # stays in same state or goes to n + 1 mod 4 with 0.5 chance
        if coin > trans_mat[int(state)][int(state)]:
            state = (state + 1) % 4

    l = np.zeros((T, 4))
    for c in range(4):
        l[:, c] = (states == c).astype(int)

    return observations, l


# Create training dataset,
# Ns: number of PCG recordings, T: length of each sound
# patch_size: patch size
def create_dataset_HMMGMM_vectors(Ns, T, nch, trans_mat, m_emis, s_emis, sigma_noise):
    sounds = np.zeros((Ns * T, nch, 1))  # -> (Ns, T, nch, 1)
    labels = np.zeros((Ns * T, 4))
    for j in range(Ns):
        new_x, new_s = synthetic_HMMGMM_vectors(trans_mat, m_emis, s_emis, sigma_noise=sigma_noise, T=T, nch=nch)
        sounds[j * T:(j + 1) * T, :, 0] = new_x
        labels[j * T:(j + 1) * T, :] = new_s

    return sounds, labels
