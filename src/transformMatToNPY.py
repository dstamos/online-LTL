import mat73
import numpy as np
temp_features = mat73.loadmat('./data/RTfeatures.mat')
features = temp_features.get("features")
features = np.delete(features, np.where(np.nanmean(features, 0) == 0), axis=1)
temp_labels = mat73.loadmat('./data/labels.mat')
labels = temp_labels.get("labels")

session = 864
features_Chris_large = []
confidence_Chris = []
correctness_Chris = []
common_mean = np.empty(30)
for task_idx in range(30):
    temp1 = labels[(task_idx)*session:(task_idx+1)*session, 1]  # (0 is correctness, 1 is confidence)
    temp2 = features[(task_idx)*session:(task_idx+1)*session]
    temp3 = labels[(task_idx)*session:(task_idx+1)*session, 0]
    temp3 = temp3[~np.isnan(temp1)]
    temp2 = temp2[~np.isnan(temp1)]
    temp1 = temp1[~np.isnan(temp1)]
    temp3 = temp3[~np.isnan(temp2[:, 0])]
    temp1 = temp1[~np.isnan(temp2[:, 0])]
    temp2 = temp2[~np.isnan(temp2[:, 0])]
    # temp2 = temp2 / norm(temp2, axis=0, keepdims=True)
    # temp2 = np.concatenate((np.ones((temp2.shape[0], 1)), temp2),axis=1)

    #temp1 = temp1-np.mean(temp1)
    features_Chris_large.append(temp2)
    confidence_Chris.append(temp1)
    correctness_Chris.append(temp3)

np.save('./data/confidence_Chris.npy', confidence_Chris, allow_pickle=True)
np.save('./data/correctness_Chris.npy', correctness_Chris, allow_pickle=True)
np.save('./data/features_Chris.npy', features_Chris_large, allow_pickle=True)
