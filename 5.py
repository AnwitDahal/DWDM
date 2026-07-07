import math
from collections import Counter

# Tree Node
class Node:
    def __init__(
        self,
        attribute=None,  # splitting attribute (internal node)
        label=None,  # class label (leaf node)
        is_leaf=False,
    ):
        self.attribute = attribute  # which feature to split on
        self.label = label  # majority / pure class at this node
        self.is_leaf = is_leaf
        self.children = {}  # attribute_value -> Node

    def add_child(self, value, node):
        self.children[value] = node


# Entropy & Information Gain
def entropy(labels):
    """Shannon entropy H(S) = -Σ p_i * log2(p_i)."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum(
        (c / n) * math.log2(c / n)
        for c in counts.values()
        if c > 0
    )


def information_gain(data, labels, attribute):
    """
    IG(S, A) = H(S) - Σ_v (|S_v| / |S|) * H(S_v)
    data : list of dicts (each dict is one sample)
    labels : list of class labels aligned with data
    attribute : the feature name to evaluate
    """
    n = len(data)
    base_entropy = entropy(labels)

    # Partition by attribute value
    partitions = {}
    for sample, label in zip(data, labels):
        val = sample[attribute]
        partitions.setdefault(val, []).append(label)

    weighted_entropy = sum(
        (len(subset) / n) * entropy(subset)
        for subset in partitions.values()
    )
    return base_entropy - weighted_entropy


# ID3 — Recursive Tree Builder
def id3(data, labels, attributes):
    """
    Recursively build a decision tree using the ID3 algorithm.

    Parameters
    ----------
    data : list of dicts
    labels : list of class labels
    attributes : list of remaining feature names to consider

    Returns
    -------
    Node (root of the subtree)
    """
    # ── Base cases ──────────────────────────────
    # All samples have the same label → pure leaf
    if len(set(labels)) == 1:
        return Node(label=labels[0], is_leaf=True)

    # No attributes left → majority-vote leaf
    if not attributes:
        majority = Counter(labels).most_common(1)[0][0]
        return Node(label=majority, is_leaf=True)

    # No data → leaf (shouldn't normally happen)
    if not data:
        return Node(label=None, is_leaf=True)

    # ── Choose best attribute ────────────────────
    gains = {attr: information_gain(data, labels, attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    node = Node(attribute=best_attr, label=Counter(labels).most_common(1)[0][0])

    # ── Partition and recurse ────────────────────
    partitions = {}
    for sample, label in zip(data, labels):
        val = sample[best_attr]
        if val not in partitions:
            partitions[val] = ([], [])
        partitions[val][0].append(sample)
        partitions[val][1].append(label)

    remaining_attrs = [a for a in attributes if a != best_attr]

    for value, (subset_data, subset_labels) in partitions.items():
        if not subset_data:
            # No samples for this branch → majority leaf
            majority = Counter(labels).most_common(1)[0][0]
            node.add_child(value, Node(label=majority, is_leaf=True))
        else:
            child = id3(subset_data, subset_labels, remaining_attrs)
            node.add_child(value, child)

    return node


# Prediction
def predict(tree, sample):
    """
    Traverse the tree for a single sample dict.
    Returns the predicted class label.
    """
    node = tree
    while not node.is_leaf:
        val = sample.get(node.attribute)
        if val not in node.children:
            # Unseen value → return majority label stored at this node
            return node.label
        node = node.children[val]
    return node.label


def predict_all(tree, samples):
    return [predict(tree, s) for s in samples]


# ASCII Tree Printer
def print_tree(node, indent=0, branch_label=""):
    prefix = " " * indent
    connector = f"[{branch_label}] " if branch_label else ""
    if node.is_leaf:
        print(f"{prefix}{connector}→ CLASS: {node.label}")
    else:
        print(f"{prefix}{connector}[{node.attribute}] (majority={node.label})")
        for value, child in sorted(node.children.items()):
            print_tree(child, indent + 1, str(value))


# Evaluation Helpers
def accuracy(y_true, y_pred):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, classes):
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def print_confusion_matrix(matrix, classes):
    print("\n Confusion Matrix (rows=actual, cols=predicted)")
    print(" " + "-" * (12 * len(classes) + 12))
    header = f" {'':>12}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    for actual in classes:
        row = f" {actual:>12}" + "".join(
            f"{matrix[actual][pred]:>12}" for pred in classes
        )
        print(row)


def print_metrics(y_true, y_pred, classes):
    print(f"\n Accuracy: {accuracy(y_true, y_pred):.2%}")
    cm = confusion_matrix(y_true, y_pred, classes)
    print_confusion_matrix(cm, classes)

    print("\n Per-class metrics:")
    print(f" {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(" " + "-" * 48)
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in classes if other != cls)
        fn = sum(cm[cls][other] for other in classes if other != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0)
        print(f" {cls:<15} {precision:>10.2%} {recall:>10.2%} {f1:>10.2%}")


# Entropy / IG Diagnostics
def print_gain_table(data, labels, attributes):
    print("\n Information Gain per attribute (root split):")
    print(f" {'Attribute':<20} {'Entropy Reduction':>18}")
    print(" " + "-" * 42)
    base_h = entropy(labels)
    print(f" {'[Dataset entropy]':<20} {base_h:>18.4f} bits")
    for attr in attributes:
        ig = information_gain(data, labels, attr)
        print(f" {attr:<20} {ig:>18.4f} bits")


# Main — Play-Tennis Dataset (classic ID3 example)
if __name__ == "__main__":
    # Classic "Play Tennis" dataset (Mitchell, 1997)
    dataset = [
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "No"},
        {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
        {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"},
    ]

    attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
    label_key = "PlayTennis"

    data = [{k: v for k, v in row.items() if k != label_key} for row in dataset]
    labels = [row[label_key] for row in dataset]
    classes = sorted(set(labels))

    print("\n" + "=" * 55)
    print(" ID3 DECISION TREE — Play Tennis")
    print("=" * 55)
    print(f" Samples : {len(data)}")
    print(f" Attributes : {attributes}")
    print(f" Classes : {classes}")

    # ── Information Gain diagnostics ────────────
    print_gain_table(data, labels, attributes)

    # ── Build tree ──────────────────────────────
    tree = id3(data, labels, attributes)

    print("\n" + "=" * 55)
    print(" DECISION TREE STRUCTURE")
    print("=" * 55)
    print_tree(tree)

    # ── Training accuracy ───────────────────────
    y_pred = predict_all(tree, data)
    print_metrics(labels, y_pred, classes)

    # ── Predict new samples ─────────────────────
    test_samples = [
        {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak"},
    ]

    print("\n" + "=" * 55)
    print(" PREDICTIONS ON NEW SAMPLES")
    print("=" * 55)
    print(f" {'Outlook':<10} {'Temp':<8} {'Humidity':<10} {'Wind':<8} {'→ Prediction'}")
    print(" " + "-" * 52)
    for s in test_samples:
        pred = predict(tree, s)
        print(
            f" {s['Outlook']:<10} {s['Temperature']:<8} "
            f"{s['Humidity']:<10} {s['Wind']:<8} → {pred}"
        )
    print()