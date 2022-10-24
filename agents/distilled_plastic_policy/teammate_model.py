import pickle


class TeammateModel:
    def __init__(self, classifier):
        self.classifier = classifier

    def action_probability(self, state, action):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        probs = self.classifier.predict_proba(state)
        return probs[0][action]

    @classmethod
    def load(cls, classifier_path):
        with open(classifier_path, 'rb') as classifier_file:
            loaded_classifier = pickle.load(classifier_file)
        return cls(loaded_classifier)

