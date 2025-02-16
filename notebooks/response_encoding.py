from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import math
from collections import Counter


class CategoricalMeanValueReplacement(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self,alpha=1,X_train=None, indi_feature_name=None,depen_feature_name=None):
        self.X_train = X_train

        self.indi_feature_name = indi_feature_name

        self.depen_feature_name = depen_feature_name

        self.unique_categories_value_counts = self.X_train[self.indi_feature_name].value_counts()

        self.giv_dict = {}
        
        self.number_unique_target_class = self.X_train[self.depen_feature_name].unique().shape[0]

        for cat, counts in self.unique_categories_value_counts.items():
            vec = []
            for k in range(1, 10):
                cls_cnt = self.X_train[(self.X_train[self.indi_feature_name] == cat) & (self.X_train[self.depen_feature_name] == k)].shape[0]
                total_count = len(self.X_train[self.X_train[self.indi_feature_name] == cat])
            
                if total_count == 0:  # Prevent division by zero
                    prob = 0
                else:
                    prob = (cls_cnt + alpha * 10) / (total_count + 90)
            
                vec.append(prob)  # Store probability for class `k`
        
            self.giv_dict[cat] = vec  # Store for each category
    
        return self


    def transform(self,X_train):
        self.response_encoding = []
        for _, row in X_train.iterrows():
            cat = row[self.indi_feature_name]
            class_idx = int(row["Class"]) - 1  # Convert to 0-based index
        
            encoding = np.zeros(self.number_unique_target_class)  # Create fresh array each time
            if cat in self.giv_dict:
                encoding[class_idx] = self.giv_dict[cat][class_idx]  # Assign probability
        
            self.response_encoding.append(encoding)
        
        return self.response_encoding


class TextMeanValueReplacement(BaseEstimator, TransformerMixin):
    def __init__(self, xtrain, feature, target, alpha, n_class, vectorizer):
        super().__init__()

        self.xtrain = xtrain
        self.feature = feature
        self.target = target
        self.alpha = alpha
        self.n_class = n_class
        self.vectorizer = vectorizer

    @staticmethod
    def get_target_probs(xtrain, target, n_class):
        probs_target = [0] * n_class
        target_counts = xtrain[target].value_counts(normalize=True).to_dict()

        for cls, probability in target_counts.items():
            probs_target[int(cls) - 1] = math.log(probability)
        
        return probs_target

    def fit(self):
        self.prob_word_per_class = [{} for _ in range(self.n_class)]
        self.target_probs = self.get_target_probs(self.xtrain, self.target, self.n_class)

        self.vectorizer.fit(self.xtrain[self.feature].tolist())
        self.vocab_size = len(self.vectorizer.vocabulary_)

        for cls in self.xtrain[self.target].unique():
            cls_index = int(cls) - 1  # Convert class label to index
            cls_df = self.xtrain[self.xtrain[self.target] == cls]
            cls_text_split = [txt.split() for txt in cls_df[self.feature].tolist()]

            word_counts = Counter(word for sublist in cls_text_split for word in sublist)
            total_word_count = sum(word_counts.values())  # Total word occurrences in the class
            unique_word_dict = {}

            for word, count in word_counts.items():
                # Apply Laplace Smoothing (Alpha-Smoothing)
                w_prob = (count + self.alpha) / (total_word_count + self.alpha * self.vocab_size)
                unique_word_dict[word] = math.log(w_prob)

            # Default probability for unseen words
            self.default_word_prob = math.log(self.alpha / (total_word_count + self.alpha * self.vocab_size))

            self.prob_word_per_class[cls_index] = unique_word_dict
        
        return self

    def transform(self, df, target):
        response_encoding = []

        for cls in df[target].unique():
            cls_index = int(cls) - 1  
            unique_word_dict = self.prob_word_per_class[cls_index]
            cls_df = df[df[target] == cls]

            cls_text_split = [txt.split() for txt in cls_df[self.feature].tolist()]

            for sublist in cls_text_split:
                word_probs = [
                    unique_word_dict.get(word, self.default_word_prob)  # Use smoothed probability for unseen words
                    for word in sublist
                ]
                prob_lxt_given_cls = sum(word_probs) + self.target_probs[cls_index]

                encoding = [0] * self.n_class
                encoding[cls_index] = prob_lxt_given_cls
                response_encoding.append(encoding)
        
        return np.abs(np.array(response_encoding))