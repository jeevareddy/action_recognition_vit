from sklearn.model_selection import train_test_split


class TrainTestSplit:

    @staticmethod
    def split(videos_paths, labels, RANDOM_STATE, DISTRIBUTED_SPLIT=False, test_size=0.2):
        if not DISTRIBUTED_SPLIT:
            return train_test_split(
                videos_paths, labels, test_size=test_size, random_state=RANDOM_STATE)

        train_video_paths = []
        test_video_paths = []
        train_labels = []
        test_labels = []

        unique_labels = set(labels)
        for action in unique_labels:
            action_paths = []
            action_labels = []
            for idx, label in enumerate(labels):
                if label == action:
                    action_paths.append(videos_paths[idx])
                    action_labels.append(action)
            train_action_path, test_action_path, train_action_labels, test_action_labels = train_test_split(
                action_paths, action_labels, test_size=test_size, random_state=RANDOM_STATE)
            train_video_paths.extend(train_action_path)
            train_labels.extend(train_action_labels)
            test_video_paths.extend(test_action_path)
            test_labels.extend(test_action_labels)

        # Return the equally splitted data
        return train_video_paths, test_video_paths, train_labels, test_labels
