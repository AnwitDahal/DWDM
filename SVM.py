import math
import random
from collections import Counter

# Kernel Functions
def kernel_linear(x1, x2):
    """K(x1,x2) = x1·x2"""
    return sum(a * b for a, b in zip(x1, x2))

def kernel_rbf(gamma=0.5):
    """K(x1,x2) = exp(−γ ‖x1−x2‖²)"""
    def k(x1, x2):
        dist_sq = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-gamma * dist_sq)
    return k

def kernel_poly(degree=3, coef=1.0):
    """K(x1,x2) = (x1·x2 + coef)^degree"""
    def k(x1, x2):
        dot = sum(a * b for a, b in zip(x1, x2))
        return (dot + coef) ** degree
    return k

def kernel_sigmoid(gamma=0.01, coef=0.0):
    """K(x1,x2) = tanh(γ x1·x2 + coef)"""
    def k(x1, x2):
        dot = sum(a * b for a, b in zip(x1, x2))
        return math.tanh(gamma * dot + coef)
    return k

# Core SVM — Dual via Sequential Minimal Optimization (SMO-lite)
class SVM:
    """
    Binary SVM solver using a simplified SMO algorithm.
    
    Parameters
    ----------
    C : regularisation (larger → harder margin, may overfit)
    kernel : callable K(x1, x2), default linear
    tol : KKT tolerance
    max_iter : max passes over the dataset without improvement
    """
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=200):
        self.C = C
        self.kernel = kernel if kernel else kernel_linear
        self.tol = tol
        self.max_iter = max_iter
        
        # Set after fit()
        self.alphas_ = None  # Lagrange multipliers
        self.b_ = 0.0  # bias
        self.X_train_ = None
        self.y_train_ = None
        self.K_cache_ = None  # kernel matrix cache
        self.support_indices_ = None

    # ── Training ──────────────────────────────
    def fit(self, X, y):
        """
        X : list of lists (n_samples × n_features)
        y : list of {+1, −1}
        """
        n = len(X)
        self.X_train_ = X
        self.y_train_ = y
        self.alphas_ = [0.0] * n
        self.b_ = 0.0
        
        # Pre-compute full kernel matrix
        self.K_cache_ = [[self.kernel(X[i], X[j]) for j in range(n)] 
                        for i in range(n)]
        
        passes = 0
        while passes < self.max_iter:
            num_changed = 0
            for i in range(n):
                Ei = self._decision_raw(i) - y[i]
                
                # Check KKT condition for sample i
                if not self._violates_kkt(Ei, self.alphas_[i], y[i]):
                    continue
                
                # Choose j ≠ i (prefer samples violating KKT)
                j = self._pick_j(i, n)
                Ej = self._decision_raw(j) - y[j]
                
                alpha_i_old = self.alphas_[i]
                alpha_j_old = self.alphas_[j]
                
                # Compute bounds L, H for alpha_j
                L, H = self._compute_bounds(y[i], y[j], alpha_i_old, alpha_j_old)
                if abs(H - L) < 1e-12:
                    continue
                
                # Compute eta (second-order optimality)
                eta = (2 * self.K_cache_[i][j] 
                       - self.K_cache_[i][i] 
                       - self.K_cache_[j][j])
                if eta >= 0:
                    continue
                
                # Update alpha_j
                self.alphas_[j] -= y[j] * (Ei - Ej) / eta
                self.alphas_[j] = self._clip(self.alphas_[j], L, H)
                
                if abs(self.alphas_[j] - alpha_j_old) < 1e-5:
                    continue
                
                # Update alpha_i (zero-sum constraint)
                self.alphas_[i] += y[i] * y[j] * (alpha_j_old - self.alphas_[j])
                
                # Update bias
                self.b_ = self._compute_bias(i, j, alpha_i_old, alpha_j_old, Ei, Ej)
                num_changed += 1
            
            passes = 0 if num_changed > 0 else passes + 1
        
        # Store only support vectors (alpha > 0)
        self.support_indices_ = [i for i in range(n) if self.alphas_[i] > 1e-5]
        return self

    # ── Internal helpers ──────────────────────
    def _decision_raw(self, i):
        """f(xᵢ) = Σⱼ αⱼ yⱼ K(xⱼ,xᵢ) + b"""
        val = 0.0
        for j in range(len(self.alphas_)):
            if self.alphas_[j] > 1e-10:
                val += self.alphas_[j] * self.y_train_[j] * self.K_cache_[j][i]
        return val + self.b_

    def _violates_kkt(self, E, alpha, yi):
        """Returns True if sample violates KKT conditions within tolerance."""
        r = E * yi
        return (r < -self.tol and alpha < self.C) or (r > self.tol and alpha > 0)

    def _pick_j(self, i, n):
        """Pick j ≠ i, prefer samples that violate KKT."""
        # First try to find a sample that violates KKT
        for _ in range(100):
            j = random.randint(0, n - 1)
            if j != i:
                Ej = self._decision_raw(j) - self.y_train_[j]
                if self._violates_kkt(Ej, self.alphas_[j], self.y_train_[j]):
                    return j
        
        # If none found, pick random
        j = i
        while j == i:
            j = random.randint(0, n - 1)
        return j

    def _compute_bounds(self, yi, yj, ai, aj):
        if yi != yj:
            L = max(0, aj - ai)
            H = min(self.C, self.C + aj - ai)
        else:
            L = max(0, ai + aj - self.C)
            H = min(self.C, ai + aj)
        return L, H

    @staticmethod
    def _clip(value, L, H):
        return max(L, min(H, value))

    def _compute_bias(self, i, j, ai_old, aj_old, Ei, Ej):
        b1 = (self.b_ - Ei 
              - self.y_train_[i] * (self.alphas_[i] - ai_old) * self.K_cache_[i][i]
              - self.y_train_[j] * (self.alphas_[j] - aj_old) * self.K_cache_[i][j])
        b2 = (self.b_ - Ej 
              - self.y_train_[i] * (self.alphas_[i] - ai_old) * self.K_cache_[i][j]
              - self.y_train_[j] * (self.alphas_[j] - aj_old) * self.K_cache_[j][j])
        
        if 0 < self.alphas_[i] < self.C:
            return b1
        elif 0 < self.alphas_[j] < self.C:
            return b2
        else:
            return (b1 + b2) / 2

    # ── Prediction ────────────────────────────
    def decision_function(self, X):
        """Raw signed distance from the hyperplane for each sample."""
        scores = []
        for x in X:
            val = 0.0
            for i in self.support_indices_:
                val += (self.alphas_[i] * self.y_train_[i] * 
                       self.kernel(self.X_train_[i], x))
            scores.append(val + self.b_)
        return scores

    def predict(self, X):
        return [1 if s >= 0 else -1 for s in self.decision_function(X)]


# Multi-class SVM — One-vs-Rest
class MultiClassSVM:
    """
    Trains one binary SVM per class (OvR).
    At prediction, picks the class whose SVM gives the highest
    decision score (most confident positive).
    """
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=200):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.classes_ = None
        self.models_ = {}  # class -> SVM

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        for cls in self.classes_:
            # Relabel: +1 for this class, -1 for all others
            binary_y = [1 if label == cls else -1 for label in y]
            model = SVM(C=self.C, kernel=self.kernel,
                       tol=self.tol, max_iter=self.max_iter)
            model.fit(X, binary_y)
            self.models_[cls] = model
            print(f" Trained classifier for class '{cls}' "
                  f"({sum(1 for l in binary_y if l==1)} positives, "
                  f"{sum(1 for l in binary_y if l==-1)} negatives) "
                  f"| support vectors: {len(model.support_indices_)}")
        return self

    def decision_function(self, X):
        """Returns {class: score} for each sample."""
        return [
            {cls: self.models_[cls].decision_function([x])[0]
             for cls in self.classes_}
            for x in X
        ]

    def predict(self, X):
        return [max(scores, key=scores.get) 
                for scores in self.decision_function(X)]


# Evaluation Utilities
def train_test_split(X, y, test_ratio=0.25, seed=42):
    random.seed(seed)
    idx = list(range(len(X)))
    random.shuffle(idx)
    cut = int(len(X) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return ([X[i] for i in tr], [X[i] for i in te],
            [y[i] for i in tr], [y[i] for i in te])

def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def confusion_matrix(y_true, y_pred, classes):
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def normalize(x, mean, std):
    return [(xi - m) / (s if s > 1e-9 else 1.0) 
            for xi, m, s in zip(x, mean, std)]

def compute_stats(X):
    n, d = len(X), len(X[0])
    mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    std = [math.sqrt(sum((X[i][j] - mean[j])**2 for i in range(n)) / n) 
           for j in range(d)]
    return mean, std

def print_metrics(y_true, y_pred, classes, title="Results"):
    print(f"\n{'='*58}\n {title}\n{'='*58}")
    acc = accuracy(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"\n Accuracy : {acc:.2%} ({correct}/{len(y_true)} correct)")
    
    cm = confusion_matrix(y_true, y_pred, classes)
    col_w = max(max(len(str(c)) for c in classes), 6) + 3
    
    print(f"\n Confusion matrix (rows=actual, cols=predicted)")
    print(" " + "-" * (col_w * (len(classes) + 1) + 2))
    print(f" {'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in classes))
    for actual in classes:
        row = f" {actual:>{col_w}}"
        for pred in classes:
            row += f"{cm[actual][pred]:>{col_w}}"
        print(row)
    
    print(f"\n Per-class metrics:")
    print(f" {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>9}")
    print(" " + "-" * 58)
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[o][cls] for o in classes if o != cls)
        fn = sum(cm[cls][o] for o in classes if o != cls)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
        print(f" {str(cls):<15} {prec:>10.2%} {rec:>10.2%} {f1:>10.2%} {support:>9}")

def print_support_vectors(model, X_train, y_train):
    print(f"\n Support vectors: {len(model.support_indices_)} "
          f"/ {len(X_train)} training samples")
    print(f" {'Index':>7} {'Label':>7} {'Alpha':>12} {'Features'}")
    print(" " + "-" * 55)
    for i in model.support_indices_[:8]:  # show first 8
        feats = " ".join(f"{v:6.3f}" for v in X_train[i])
        print(f" {i:>7} {y_train[i]:>7} {model.alphas_[i]:>12.6f} [{feats}]")
    if len(model.support_indices_) > 8:
        print(f" ... ({len(model.support_indices_)-8} more)")

def ascii_boundary(model, X, y, feature_names=("x1", "x2"), res=28):
    """2-D ASCII decision boundary plot (works for binary SVM)."""
    x_vals = [xi[0] for xi in X]
    y_vals = [xi[1] for xi in X]
    x_min, x_max = min(x_vals) - 0.5, max(x_vals) + 0.5
    y_min, y_max = min(y_vals) - 0.5, max(y_vals) + 0.5
    
    print(f"\n Decision boundary [{feature_names[0]} × {feature_names[1]}]")
    print(" " + "─" * (res + 4))
    
    for row in range(res // 2):
        yc = y_max - (row / (res // 2 - 1)) * (y_max - y_min)
        line = ""
        for col in range(res):
            xc = x_min + (col / (res - 1)) * (x_max - x_min)
            score = model.decision_function([[xc, yc]])[0]
            
            # Check if a training point is near this cell
            closest = min(X, key=lambda p: (p[0]-xc)**2 + (p[1]-yc)**2)
            dist = math.sqrt((closest[0]-xc)**2 + (closest[1]-yc)**2)
            if dist < (x_max - x_min) / res * 1.2:
                idx = X.index(closest)
                line += "●" if y[idx] == 1 else "○"
            elif abs(score) < 0.25:
                line += "│"  # near the boundary
            elif score > 0:
                line += "·"
            else:
                line += " "
        print(f" |{line}|")
    print(" " + "─" * (res + 4))
    print(f" Legend: ● = class +1 ○ = class -1 │ = boundary")


# Demo Datasets
def make_linearly_separable(n=40, seed=0):
    """Two Gaussian blobs, linearly separable."""
    random.seed(seed)
    X, y = [], []
    for _ in range(n // 2):
        X.append([random.gauss(2.0, 0.6), random.gauss(2.0, 0.6)])
        y.append(1)
    for _ in range(n // 2):
        X.append([random.gauss(-2.0, 0.6), random.gauss(-2.0, 0.6)])
        y.append(-1)
    return X, y

def make_xor(n=60, seed=1):
    """XOR pattern — not linearly separable, needs kernel."""
    random.seed(seed)
    X, y = [], []
    for _ in range(n // 4):
        X.append([random.gauss(1.5, 0.4), random.gauss(1.5, 0.4)]); y.append(1)
        X.append([random.gauss(-1.5, 0.4), random.gauss(-1.5, 0.4)]); y.append(1)
        X.append([random.gauss(1.5, 0.4), random.gauss(-1.5, 0.4)]); y.append(-1)
        X.append([random.gauss(-1.5, 0.4), random.gauss(1.5, 0.4)]); y.append(-1)
    return X, y

def make_iris_3class():
    """Abbreviated Iris dataset — 3 classes, 4 features."""
    data = [
        ([5.1,3.5,1.4,0.2],"setosa"),([4.9,3.0,1.4,0.2],"setosa"),
        ([4.7,3.2,1.3,0.2],"setosa"),([4.6,3.1,1.5,0.2],"setosa"),
        ([5.0,3.6,1.4,0.2],"setosa"),([5.4,3.9,1.7,0.4],"setosa"),
        ([4.6,3.4,1.4,0.3],"setosa"),([5.0,3.4,1.5,0.2],"setosa"),
        ([4.9,3.1,1.5,0.1],"setosa"),([5.4,3.7,1.5,0.2],"setosa"),
        ([4.8,3.4,1.6,0.2],"setosa"),([5.8,4.0,1.2,0.2],"setosa"),
        ([5.7,4.4,1.5,0.4],"setosa"),([5.4,3.9,1.3,0.4],"setosa"),
        ([5.1,3.5,1.4,0.3],"setosa"),([5.7,3.8,1.7,0.3],"setosa"),
        ([5.1,3.8,1.5,0.3],"setosa"),([5.4,3.4,1.7,0.2],"setosa"),
        ([5.1,3.7,1.5,0.4],"setosa"),([4.6,3.6,1.0,0.2],"setosa"),
        ([7.0,3.2,4.7,1.4],"versicolor"),([6.4,3.2,4.5,1.5],"versicolor"),
        ([6.9,3.1,4.9,1.5],"versicolor"),([5.5,2.3,4.0,1.3],"versicolor"),
        ([6.5,2.8,4.6,1.5],"versicolor"),([5.7,2.8,4.5,1.3],"versicolor"),
        ([6.3,3.3,4.7,1.6],"versicolor"),([4.9,2.4,3.3,1.0],"versicolor"),
        ([6.6,2.9,4.6,1.3],"versicolor"),([5.2,2.7,3.9,1.4],"versicolor"),
        ([5.9,3.0,4.2,1.5],"versicolor"),([6.0,2.2,4.0,1.0],"versicolor"),
        ([6.1,2.9,4.7,1.4],"versicolor"),([5.6,2.9,3.6,1.3],"versicolor"),
        ([6.7,3.1,4.4,1.4],"versicolor"),([5.6,3.0,4.5,1.5],"versicolor"),
        ([5.8,2.7,4.1,1.0],"versicolor"),([6.2,2.2,4.5,1.5],"versicolor"),
        ([5.6,2.5,3.9,1.1],"versicolor"),([5.9,3.2,4.8,1.8],"versicolor"),
        ([6.3,3.3,6.0,2.5],"virginica"), ([5.8,2.7,5.1,1.9],"virginica"),
        ([7.1,3.0,5.9,2.1],"virginica"), ([6.3,2.9,5.6,1.8],"virginica"),
        ([6.5,3.0,5.8,2.2],"virginica"), ([7.6,3.0,6.6,2.1],"virginica"),
        ([7.3,2.9,6.3,1.8],"virginica"), ([6.7,2.5,5.8,1.8],"virginica"),
        ([7.2,3.6,6.1,2.5],"virginica"), ([6.5,3.2,5.1,2.0],"virginica"),
        ([6.4,2.7,5.3,1.9],"virginica"), ([6.8,3.0,5.5,2.1],"virginica"),
        ([5.7,2.5,5.0,2.0],"virginica"), ([5.8,2.8,5.1,2.4],"virginica"),
        ([6.4,3.2,5.3,2.3],"virginica"), ([6.5,3.0,5.5,1.8],"virginica"),
        ([7.7,3.8,6.7,2.2],"virginica"), ([7.7,2.6,6.9,2.3],"virginica"),
        ([6.0,2.2,5.0,1.5],"virginica"), ([6.9,3.2,5.7,2.3],"virginica"),
    ]
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    return X, y


# Main
if __name__ == "__main__":
    random.seed(42)
    
    # ── Demo 1: Linear SVM — linearly separable blobs ──
    print("\n" + "="*58)
    print(" DEMO 1 — Linear SVM (linearly separable blobs)")
    print("="*58)
    X_lin, y_lin = make_linearly_separable(n=50, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X_lin, y_lin, 0.25, seed=42)
    
    mean, std = compute_stats(X_tr)
    X_tr_n = [normalize(x, mean, std) for x in X_tr]
    X_te_n = [normalize(x, mean, std) for x in X_te]
    
    svm_lin = SVM(C=1.0, kernel=kernel_linear, max_iter=300)
    svm_lin.fit(X_tr_n, y_tr)
    y_pred_lin = svm_lin.predict(X_te_n)
    
    print_metrics(y_te, y_pred_lin, [-1, 1], "Linear SVM — Test Results")
    print_support_vectors(svm_lin, X_tr_n, y_tr)
    ascii_boundary(svm_lin, X_tr_n, y_tr)
    
    # ── Demo 2: RBF Kernel SVM — XOR (non-linear) ──────
    print("\n" + "="*58)
    print(" DEMO 2 — RBF Kernel SVM (XOR — non-linear data)")
    print("="*58)
    X_xor, y_xor = make_xor(n=60, seed=1)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_xor, y_xor, 0.25, seed=0)
    
    mean2, std2 = compute_stats(X_tr2)
    X_tr2_n = [normalize(x, mean2, std2) for x in X_tr2]
    X_te2_n = [normalize(x, mean2, std2) for x in X_te2]
    
    svm_rbf = SVM(C=5.0, kernel=kernel_rbf(gamma=0.8), max_iter=300)
    svm_rbf.fit(X_tr2_n, y_tr2)
    y_pred_rbf = svm_rbf.predict(X_te2_n)
    
    print_metrics(y_te2, y_pred_rbf, [-1, 1], "RBF Kernel SVM — Test Results")
    print_support_vectors(svm_rbf, X_tr2_n, y_tr2)
    ascii_boundary(svm_rbf, X_tr2_n, y_tr2)
    
    # ── Demo 3: Kernel comparison ───────────────────────
    print("\n" + "="*58)
    print(" DEMO 3 — Kernel Comparison (XOR dataset)")
    print("="*58)
    
    kernels = {
        "Linear ": kernel_linear,
        "RBF γ=0.8 ": kernel_rbf(gamma=0.8),
        "Poly d=3 ": kernel_poly(degree=3),
        "Sigmoid ": kernel_sigmoid(gamma=0.5, coef=0.0),
    }
    
    print(f"\n {'Kernel':<18} {'Train Acc':>10} {'Test Acc':>10} {'SVs':>6}")
    print(" " + "-" * 48)
    for name, kfn in kernels.items():
        m = SVM(C=5.0, kernel=kfn, max_iter=200)
        m.fit(X_tr2_n, y_tr2)
        tr_acc = accuracy(y_tr2, m.predict(X_tr2_n))
        te_acc = accuracy(y_te2, m.predict(X_te2_n))
        print(f" {name:<18} {tr_acc:>10.2%} {te_acc:>10.2%} {len(m.support_indices_):>6}")
    
    # ── Demo 4: Multi-class SVM (OvR) — Iris ───────────
    print("\n" + "="*58)
    print(" DEMO 4 — Multi-class SVM OvR (Iris, 3 classes)")
    print("="*58)
    X_iris, y_iris = make_iris_3class()
    X_tr4, X_te4, y_tr4, y_te4 = train_test_split(X_iris, y_iris, 0.25, seed=5)
    
    mean4, std4 = compute_stats(X_tr4)
    X_tr4_n = [normalize(x, mean4, std4) for x in X_tr4]
    X_te4_n = [normalize(x, mean4, std4) for x in X_te4]
    
    print("\n Training one-vs-rest classifiers...")
    mc_svm = MultiClassSVM(C=2.0, kernel=kernel_rbf(gamma=0.5), max_iter=200)
    mc_svm.fit(X_tr4_n, y_tr4)
    y_pred4 = mc_svm.predict(X_te4_n)
    
    print_metrics(y_te4, y_pred4, mc_svm.classes_, "Multi-class SVM — Test Results")
    
    # Show decision scores for first 5 test samples
    print("\n Decision scores for first 5 test samples:")
    print(f" {'True':>12} {'Pred':>12} " +
          " ".join(f"{c:>12}" for c in mc_svm.classes_))
    print(" " + "-" * 65)
    scores_all = mc_svm.decision_function(X_te4_n[:5])
    for true, pred, scores in zip(y_te4[:5], y_pred4[:5], scores_all):
        row = f" {true:>12} {pred:>12} "
        row += " ".join(f"{scores[c]:>12.4f}" for c in mc_svm.classes_)
        match = "✓" if true == pred else "✗"
        print(row + f" {match}")
    print()