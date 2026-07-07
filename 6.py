import math
from collections import defaultdict, Counter

# 1. GAUSSIAN NAIVE BAYES (continuous features)
class GaussianNB:
    """
    For each class c and feature j, model P(x_j | c) as a Gaussian:
    N(x; μ_cj, σ_cj)
    At prediction time:
    log P(c | x) ∝ log P(c) + Σ_j log N(x_j; μ_cj, σ_cj)
    """
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing  # prevents division by zero
        self.classes_ = None
        self.class_prior_ = {}  # P(c)
        self.mean_ = {}  # μ_cj
        self.var_ = {}  # σ²_cj
        self.class_counts_ = {}  # Store class counts for smoothing

    # ── Training ──────────────────────────────
    def fit(self, X, y):
        """
        X : list of lists [[f1, f2, ...], ...]
        y : list of labels
        """
        n_samples = len(X)
        n_features = len(X[0])
        self.classes_ = sorted(set(y))

        for cls in self.classes_:
            # Collect all samples for this class
            X_cls = [X[i] for i in range(n_samples) if y[i] == cls]
            n_cls = len(X_cls)
            
            self.class_prior_[cls] = n_cls / n_samples
            self.class_counts_[cls] = n_cls
            
            # Per-feature mean and variance
            self.mean_[cls] = []
            self.var_[cls] = []
            
            for j in range(n_features):
                col = [X_cls[i][j] for i in range(n_cls)]
                mean = sum(col) / n_cls
                var = sum((x - mean) ** 2 for x in col) / n_cls
                self.mean_[cls].append(mean)
                self.var_[cls].append(var + self.var_smoothing)
        
        return self

    # ── Prediction ────────────────────────────
    def _log_likelihood(self, cls, x):
        """Σ_j log N(x_j; μ_cj, σ²_cj)"""
        log_prob = 0.0
        for j, xj in enumerate(x):
            mu = self.mean_[cls][j]
            var = self.var_[cls][j]
            # Correct Gaussian log-likelihood
            log_prob += -0.5 * math.log(2 * math.pi * var)
            log_prob += -((xj - mu) ** 2) / (2 * var)
        return log_prob

    def predict_log_proba(self, X):
        """Return log-posterior for each class (unnormalized)."""
        results = []
        for x in X:
            scores = {
                cls: math.log(self.class_prior_[cls]) + self._log_likelihood(cls, x)
                for cls in self.classes_
            }
            results.append(scores)
        return results

    def predict_proba(self, X):
        """Return normalized posterior probabilities."""
        log_scores = self.predict_log_proba(X)
        results = []
        for scores in log_scores:
            max_log = max(scores.values())
            exp_scores = {cls: math.exp(v - max_log) for cls, v in scores.items()}
            total = sum(exp_scores.values())
            results.append({cls: v / total for cls, v in exp_scores.items()})
        return results

    def predict(self, X):
        return [max(scores, key=scores.get) for scores in self.predict_proba(X)]


# 2. CATEGORICAL NAIVE BAYES (discrete features)
class CategoricalNB:
    """
    For each class c, feature j, value v:
    P(x_j = v | c) = (count(x_j=v, c) + α) / (count(c) + α * |V_j|)
    α = Laplace smoothing parameter (default 1).
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = {}
        self.class_counts_ = {}  # Store class counts
        self.feature_prob_ = {}  # [cls][feature_idx][value] = log P
        self.vocab_ = []  # list of sets of unique values per feature

    def fit(self, X, y):
        """
        X : list of lists of categorical values [['Sunny','Hot','High'], ...]
        y : list of labels
        """
        n_samples = len(X)
        n_features = len(X[0])
        self.classes_ = sorted(set(y))

        # Discover unique values per feature
        self.vocab_ = [sorted(set(X[i][j] for i in range(n_samples)))
                      for j in range(n_features)]

        for cls in self.classes_:
            X_cls = [X[i] for i in range(n_samples) if y[i] == cls]
            n_cls = len(X_cls)
            
            self.class_prior_[cls] = n_cls / n_samples
            self.class_counts_[cls] = n_cls
            self.feature_prob_[cls] = []
            
            for j in range(n_features):
                col = [X_cls[i][j] for i in range(n_cls)]
                counts = Counter(col)
                vocab_j = self.vocab_[j]
                
                # Laplace-smoothed log probabilities
                log_probs = {
                    v: math.log((counts.get(v, 0) + self.alpha) /
                                (n_cls + self.alpha * len(vocab_j)))
                    for v in vocab_j
                }
                self.feature_prob_[cls].append(log_probs)
        
        return self

    def predict_proba(self, X):
        results = []
        for x in X:
            scores = {}
            for cls in self.classes_:
                log_p = math.log(self.class_prior_[cls])
                n_cls = self.class_counts_[cls]  # Get class count from stored value
                for j, val in enumerate(x):
                    # Use smoothed fallback for unseen values
                    log_p += self.feature_prob_[cls][j].get(
                        val, math.log(self.alpha / (n_cls + self.alpha * len(self.vocab_[j])))
                    )
                scores[cls] = log_p
            
            # Normalize via log-sum-exp
            max_log = max(scores.values())
            exp_scores = {cls: math.exp(v - max_log) for cls, v in scores.items()}
            total = sum(exp_scores.values())
            results.append({cls: v / total for cls, v in exp_scores.items()})
        return results

    def predict(self, X):
        return [max(scores, key=scores.get) for scores in self.predict_proba(X)]


# 3. MULTINOMIAL NAIVE BAYES (text / counts)
class MultinomialNB:
    """
    For text classification using word-count vectors.
    P(w | c) = (count(w, c) + α) / (total_words_in_c + α * |vocab|)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = {}
        self.class_counts_ = {}  # Store class counts
        self.log_feat_prob_ = {}  # cls -> list of log P(w_j | cls)
        self.vocab_size_ = 0

    def fit(self, X, y):
        """
        X : list of lists of word counts [[2, 0, 1, ...], ...]
        y : list of labels
        """
        n_samples = len(X)
        self.vocab_size_ = len(X[0])
        self.classes_ = sorted(set(y))

        for cls in self.classes_:
            X_cls = [X[i] for i in range(n_samples) if y[i] == cls]
            n_cls = len(X_cls)
            self.class_prior_[cls] = n_cls / n_samples
            self.class_counts_[cls] = n_cls
            
            # Sum word counts across all documents in this class
            word_counts = [sum(X_cls[i][j] for i in range(len(X_cls)))
                          for j in range(self.vocab_size_)]
            total = sum(word_counts)
            
            self.log_feat_prob_[cls] = [
                math.log((wc + self.alpha) / (total + self.alpha * self.vocab_size_))
                for wc in word_counts
            ]
        
        return self

    def predict_proba(self, X):
        results = []
        for x in X:
            scores = {}
            for cls in self.classes_:
                log_p = math.log(self.class_prior_[cls])
                log_p += sum(x[j] * self.log_feat_prob_[cls][j]
                             for j in range(self.vocab_size_))
                scores[cls] = log_p
            
            max_log = max(scores.values())
            exp_scores = {cls: math.exp(v - max_log) for cls, v in scores.items()}
            total = sum(exp_scores.values())
            results.append({cls: v / total for cls, v in exp_scores.items()})
        
        return results

    def predict(self, X):
        return [max(scores, key=scores.get) for scores in self.predict_proba(X)]


# Utilities
def train_test_split(X, y, test_ratio=0.3, seed=42):
    """Simple deterministic train/test split."""
    import random
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    cut = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:cut], indices[cut:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred, classes):
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def print_metrics(y_true, y_pred, classes, title=""):
    if title:
        print(f"\n{'='*55}\n {title}\n{'='*55}")
    
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, classes)
    col_w = max(len(str(c)) for c in classes) + 4
    
    print(f"\n Accuracy : {acc:.2%} ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)} correct)")
    print(f"\n Confusion matrix (rows=actual, cols=predicted)")
    header = f" {'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in classes)
    print(" " + "-" * (col_w * (len(classes)+1)))
    print(header)
    for actual in classes:
        row = f" {actual:>{col_w}}" + "".join(
            f"{cm[actual][pred]:>{col_w}}" for pred in classes)
        print(row)
    
    print(f"\n Per-class metrics:")
    print(f" {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(" " + "-" * 50)
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[o][cls] for o in classes if o != cls)
        fn = sum(cm[cls][o] for o in classes if o != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2*precision*recall/(precision+recall)
              if (precision+recall) else 0.0)
        print(f" {str(cls):<15} {precision:>10.2%} {recall:>10.2%} {f1:>10.2%}")


def print_posterior(sample_label, proba_dict):
    total_bar = 30
    print(f"\n Sample: {sample_label}")
    for cls, prob in sorted(proba_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * total_bar)
        print(f" {str(cls):<12} {prob:>6.2%} {bar}")


# Demo 1 — GaussianNB on Iris-like data
def demo_gaussian():
    print("\n" + "=" * 55)
    print(" DEMO 1 — Gaussian Naive Bayes (Iris dataset)")
    print("=" * 55)
    
    # Abbreviated Iris dataset [sepal_len, sepal_wid, petal_len, petal_wid]
    data = [
        ([5.1,3.5,1.4,0.2],"setosa"),([4.9,3.0,1.4,0.2],"setosa"),
        ([4.7,3.2,1.3,0.2],"setosa"),([4.6,3.1,1.5,0.2],"setosa"),
        ([5.0,3.6,1.4,0.2],"setosa"),([5.4,3.9,1.7,0.4],"setosa"),
        ([4.6,3.4,1.4,0.3],"setosa"),([5.0,3.4,1.5,0.2],"setosa"),
        ([4.4,2.9,1.4,0.2],"setosa"),([4.9,3.1,1.5,0.1],"setosa"),
        ([5.4,3.7,1.5,0.2],"setosa"),([4.8,3.4,1.6,0.2],"setosa"),
        ([4.8,3.0,1.4,0.1],"setosa"),([4.3,3.0,1.1,0.1],"setosa"),
        ([5.8,4.0,1.2,0.2],"setosa"),([5.7,4.4,1.5,0.4],"setosa"),
        ([5.4,3.9,1.3,0.4],"setosa"),([5.1,3.5,1.4,0.3],"setosa"),
        ([5.7,3.8,1.7,0.3],"setosa"),([5.1,3.8,1.5,0.3],"setosa"),
        ([7.0,3.2,4.7,1.4],"versicolor"),([6.4,3.2,4.5,1.5],"versicolor"),
        ([6.9,3.1,4.9,1.5],"versicolor"),([5.5,2.3,4.0,1.3],"versicolor"),
        ([6.5,2.8,4.6,1.5],"versicolor"),([5.7,2.8,4.5,1.3],"versicolor"),
        ([6.3,3.3,4.7,1.6],"versicolor"),([4.9,2.4,3.3,1.0],"versicolor"),
        ([6.6,2.9,4.6,1.3],"versicolor"),([5.2,2.7,3.9,1.4],"versicolor"),
        ([5.0,2.0,3.5,1.0],"versicolor"),([5.9,3.0,4.2,1.5],"versicolor"),
        ([6.0,2.2,4.0,1.0],"versicolor"),([6.1,2.9,4.7,1.4],"versicolor"),
        ([5.6,2.9,3.6,1.3],"versicolor"),([6.7,3.1,4.4,1.4],"versicolor"),
        ([5.6,3.0,4.5,1.5],"versicolor"),([5.8,2.7,4.1,1.0],"versicolor"),
        ([6.2,2.2,4.5,1.5],"versicolor"),([5.6,2.5,3.9,1.1],"versicolor"),
        ([6.3,3.3,6.0,2.5],"virginica"), ([5.8,2.7,5.1,1.9],"virginica"),
        ([7.1,3.0,5.9,2.1],"virginica"), ([6.3,2.9,5.6,1.8],"virginica"),
        ([6.5,3.0,5.8,2.2],"virginica"), ([7.6,3.0,6.6,2.1],"virginica"),
        ([4.9,2.5,4.5,1.7],"virginica"), ([7.3,2.9,6.3,1.8],"virginica"),
        ([6.7,2.5,5.8,1.8],"virginica"), ([7.2,3.6,6.1,2.5],"virginica"),
        ([6.5,3.2,5.1,2.0],"virginica"), ([6.4,2.7,5.3,1.9],"virginica"),
        ([6.8,3.0,5.5,2.1],"virginica"), ([5.7,2.5,5.0,2.0],"virginica"),
        ([5.8,2.8,5.1,2.4],"virginica"), ([6.4,3.2,5.3,2.3],"virginica"),
        ([6.5,3.0,5.5,1.8],"virginica"), ([7.7,3.8,6.7,2.2],"virginica"),
        ([7.7,2.6,6.9,2.3],"virginica"), ([6.0,2.2,5.0,1.5],"virginica"),
    ]
    
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print_metrics(y_test, y_pred, model.classes_, "")
    
    # Show posteriors for a few test samples
    print("\n Posterior probabilities (first 4 test samples):")
    probas = model.predict_proba(X_test[:4])
    features = ["sepal_len","sepal_wid","petal_len","petal_wid"]
    for i, (proba, sample) in enumerate(zip(probas, X_test[:4])):
        label = dict(zip(features, [f"{v:.1f}" for v in sample]))
        print_posterior(label, proba)


# Demo 2 — CategoricalNB on Play-Tennis
def demo_categorical():
    print("\n" + "=" * 55)
    print(" DEMO 2 — Categorical Naive Bayes (Play Tennis)")
    print("=" * 55)
    
    dataset = [
        (["Sunny", "Hot", "High", "Weak"], "No"),
        (["Sunny", "Hot", "High", "Strong"], "No"),
        (["Overcast", "Hot", "High", "Weak"], "Yes"),
        (["Rain", "Mild", "High", "Weak"], "Yes"),
        (["Rain", "Cool", "Normal", "Weak"], "Yes"),
        (["Rain", "Cool", "Normal", "Strong"], "No"),
        (["Overcast", "Cool", "Normal", "Strong"], "Yes"),
        (["Sunny", "Mild", "High", "Weak"], "No"),
        (["Sunny", "Cool", "Normal", "Weak"], "Yes"),
        (["Rain", "Mild", "Normal", "Weak"], "Yes"),
        (["Sunny", "Mild", "Normal", "Strong"], "Yes"),
        (["Overcast", "Mild", "High", "Strong"], "Yes"),
        (["Overcast", "Hot", "Normal", "Weak"], "Yes"),
        (["Rain", "Mild", "High", "Strong"], "No"),
    ]
    
    X = [d[0] for d in dataset]
    y = [d[1] for d in dataset]
    
    model = CategoricalNB(alpha=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    print_metrics(y, y_pred, model.classes_)
    
    # New samples to classify
    test = [
        ["Sunny", "Cool", "High", "Strong"],
        ["Overcast", "Hot", "Normal", "Weak"],
        ["Rain", "Mild", "High", "Weak"],
    ]
    
    features = ["Outlook", "Temperature", "Humidity", "Wind"]
    print("\n Posteriors for new samples:")
    for sample in test:
        proba = model.predict_proba([sample])[0]
        label = dict(zip(features, sample))
        print_posterior(label, proba)


# Demo 3 — MultinomialNB on text (spam filter)
def demo_multinomial():
    print("\n" + "=" * 55)
    print(" DEMO 3 — Multinomial Naive Bayes (Spam Filter)")
    print("=" * 55)
    
    # Vocabulary
    vocab = ["free","win","prize","click","offer",
             "meeting","project","report","agenda","deadline"]
    
    # Training emails as word-count vectors [vocab order]
    # free win prize click offer meet proj rep agen dead
    spam = [
        [3,2,1,1,2,0,0,0,0,0],
        [2,3,2,0,1,0,0,0,0,0],
        [1,1,3,2,0,0,0,0,0,0],
        [0,2,1,3,2,0,0,0,0,0],
        [4,1,0,1,3,0,0,0,0,0],
        [2,0,2,2,1,0,0,0,0,0],
    ]
    
    ham = [
        [0,0,0,0,0,2,3,1,1,2],
        [0,0,0,0,0,1,2,3,0,1],
        [0,0,0,0,0,3,1,0,2,2],
        [0,0,0,0,0,0,2,2,1,3],
        [0,0,0,0,0,2,0,3,2,1],
        [0,0,0,1,0,1,2,1,0,2],
    ]
    
    X = spam + ham
    y = ["spam"] * len(spam) + ["ham"] * len(ham)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25, seed=7)
    
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print_metrics(y_test, y_pred, model.classes_)
    
    # Show learned word probabilities
    print("\n Learned P(word | class):")
    print(f" {'Word':<12}" + "".join(f" {c:>8}" for c in model.classes_))
    print(" " + "-" * 36)
    for j, word in enumerate(vocab):
        row = f" {word:<12}"
        for cls in model.classes_:
            row += f" {math.exp(model.log_feat_prob_[cls][j]):>8.4f}"
        print(row)
    
    # Classify a new email
    print("\n Classifying a new email...")
    new_email = [2, 1, 0, 3, 1, 0, 0, 0, 0, 0]  # many click/free words
    words_present = [(vocab[i], new_email[i]) for i in range(len(vocab)) if new_email[i] > 0]
    print(f" Words: {words_present}")
    proba = model.predict_proba([new_email])[0]
    print_posterior("new email", proba)


# Entry Point
if __name__ == "__main__":
    demo_gaussian()
    demo_categorical()
    demo_multinomial()
    print()