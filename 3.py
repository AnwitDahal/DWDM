from itertools import combinations
from collections import defaultdict


# Core Apriori Functions
def get_frequent_1_itemsets(transactions, min_support):
    """Scan transactions and return frequent 1-itemsets with their support."""
    item_count = defaultdict(int)
    n = len(transactions)

    for transaction in transactions:
        for item in transaction:
            item_count[frozenset([item])] += 1

    frequent = {
        itemset: count / n
        for itemset, count in item_count.items()
        if count / n >= min_support
    }

    return frequent


def generate_candidates(prev_frequent, k):
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets.
    Uses the Apriori join and pruning step.
    """
    prev_list = list(prev_frequent.keys())
    candidates = set()

    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            union = prev_list[i] | prev_list[j]

            if len(union) == k:
                # Apriori pruning
                all_subsets_frequent = all(
                    frozenset(sub) in prev_frequent
                    for sub in combinations(union, k - 1)
                )

                if all_subsets_frequent:
                    candidates.add(union)

    return candidates


def get_frequent_itemsets(transactions, min_support):
    """
    Run the full Apriori algorithm to find all frequent itemsets.
    """
    n = len(transactions)
    transaction_sets = [set(t) for t in transactions]

    all_frequent = {}

    # Pass 1
    current_frequent = get_frequent_1_itemsets(transactions, min_support)
    all_frequent.update(current_frequent)

    k = 2

    while current_frequent:
        candidates = generate_candidates(current_frequent, k)

        if not candidates:
            break

        candidate_count = defaultdict(int)

        for transaction in transaction_sets:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_count[candidate] += 1

        current_frequent = {
            itemset: count / n
            for itemset, count in candidate_count.items()
            if count / n >= min_support
        }

        all_frequent.update(current_frequent)
        k += 1

    return all_frequent


def generate_association_rules(frequent_itemsets, min_confidence):
    """
    Generate association rules from frequent itemsets.
    """
    rules = []

    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)

        for size in range(1, len(items)):
            for antecedent_tuple in combinations(items, size):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                ant_support = frequent_itemsets.get(antecedent)

                if ant_support is None:
                    continue

                confidence = support / ant_support

                if confidence < min_confidence:
                    continue

                con_support = frequent_itemsets.get(consequent)
                lift = confidence / con_support if con_support else float("inf")

                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                })

    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))

    return rules


# Pretty Printing Helpers
def print_frequent_itemsets(frequent_itemsets):
    print("\n" + "=" * 55)
    print(" FREQUENT ITEMSETS")
    print("=" * 55)

    by_size = defaultdict(list)

    for itemset, sup in frequent_itemsets.items():
        by_size[len(itemset)].append((itemset, sup))

    for k in sorted(by_size):
        print(f"\n{k}-itemsets ({len(by_size[k])} found)")
        print("-" * 40)

        for itemset, sup in sorted(by_size[k], key=lambda x: -x[1]):
            items_str = "{" + ", ".join(sorted(itemset)) + "}"
            print(f"{items_str:<30} Support = {sup:.2%}")


def print_rules(rules):
    print("\n" + "=" * 70)
    print(" ASSOCIATION RULES")
    print("=" * 70)

    if not rules:
        print("No rules found.")
        return

    print(f"{'Rule':<40} {'Support':>10} {'Confidence':>12} {'Lift':>8}")
    print("-" * 70)

    for r in rules:
        lhs = "{" + ", ".join(sorted(r["antecedent"])) + "}"
        rhs = "{" + ", ".join(sorted(r["consequent"])) + "}"

        print(
            f"{lhs} -> {rhs:<22} "
            f"{r['support']:>8.2%} "
            f"{r['confidence']:>10.2%} "
            f"{r['lift']:>8.2f}"
        )


# Main Program
if __name__ == "__main__":

    # Sample grocery transaction database
    transactions = [
        ["milk", "bread", "butter"],
        ["beer", "bread"],
        ["milk", "bread", "butter", "beer"],
        ["milk", "bread"],
        ["bread", "butter"],
        ["milk", "butter"],
        ["milk", "bread", "butter", "beer"],
        ["milk", "bread", "beer"],
        ["bread", "butter", "beer"],
        ["milk", "bread", "butter"],
    ]

    MIN_SUPPORT = 0.40
    MIN_CONFIDENCE = 0.60

    print(f"\nTransactions    : {len(transactions)}")
    print(f"Min Support     : {MIN_SUPPORT:.0%}")
    print(f"Min Confidence  : {MIN_CONFIDENCE:.0%}")

    # Step 1
    frequent_itemsets = get_frequent_itemsets(transactions, MIN_SUPPORT)
    print_frequent_itemsets(frequent_itemsets)

    # Step 2
    rules = generate_association_rules(frequent_itemsets, MIN_CONFIDENCE)
    print_rules(rules)

    print(f"\nTotal Frequent Itemsets : {len(frequent_itemsets)}")
    print(f"Total Association Rules : {len(rules)}")