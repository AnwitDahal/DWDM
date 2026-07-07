# FP-Growth Algorithm Implementation
from collections import defaultdict
from itertools import combinations

# FP-Tree Node
class FPNode:
    def __init__(self, item, count=0, parent=None):
        self.item = item  # item label (None for root)
        self.count = count  # support count through this node
        self.parent = parent  # parent FPNode
        self.children = {}  # item -> FPNode
        self.link = None  # next node with same item (header link)

    def increment(self, count=1):
        self.count += count

# FP-Tree
class FPTree:
    def __init__(self):
        self.root = FPNode(None)  # virtual root
        self.header = {}  # item -> [support, first_node]

    # ── Build ──────────────────────────────────
    def insert_transaction(self, transaction, count=1):
        """Insert a single (ordered) transaction into the tree."""
        node = self.root
        for item in transaction:
            if item in node.children:
                child = node.children[item]
                child.increment(count)
                # Update header table count
                if item in self.header:
                    self.header[item][0] += count
                node = child
            else:
                child = FPNode(item, count, parent=node)
                node.children[item] = child
                self._link_header(item, child, count)
                node = child

    def _link_header(self, item, new_node, count):
        """Append new_node to the linked list for item in the header table."""
        if item not in self.header:
            self.header[item] = [count, None]  # Initialize with count
        else:
            self.header[item][0] += count
        
        # Link the node
        if self.header[item][1] is None:
            self.header[item][1] = new_node
        else:
            current = self.header[item][1]
            while current.link:
                current = current.link
            current.link = new_node

    # ── Query ──────────────────────────────────
    def is_single_path(self):
        """Return True if the tree has only one branch (no merging needed)."""
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return False
            node = next(iter(node.children.values()))
        return True

    def nodes_for(self, item):
        """Yield all nodes in the linked list for item."""
        node = self.header.get(item, [None, None])[1]
        while node:
            yield node
            node = node.link

    def prefix_path(self, node):
        """Return the prefix path (ancestors) leading to node, without node itself."""
        path = []
        parent = node.parent
        while parent and parent.item is not None:
            path.append(parent.item)
            parent = parent.parent
        return list(reversed(path))

# Build initial FP-Tree from transactions
def build_fp_tree(transactions, min_support):
    n = len(transactions)
    freq_count = defaultdict(int)
    for txn in transactions:
        for item in txn:
            freq_count[item] += 1
    
    # Keep only items that meet min_support (as count, not ratio)
    min_count = min_support * n
    freq_items = {item: cnt for item, cnt in freq_count.items() if cnt >= min_count}
    
    tree = FPTree()
    for txn in transactions:
        # Filter and sort: most frequent first, ties broken alphabetically
        ordered = sorted(
            [item for item in txn if item in freq_items],
            key=lambda x: (-freq_items[x], x)
        )
        if ordered:
            tree.insert_transaction(ordered)
    
    item_support = {item: cnt / n for item, cnt in freq_items.items()}
    return tree, item_support, n

# Recursive FP-Growth Mining
def fp_growth(tree, min_support, n, prefix=frozenset()):
    """
    Recursively mine frequent itemsets from an FP-Tree.
    Yields (frozenset, support) for every frequent itemset found.
    """
    min_count = min_support * n
    
    if tree.is_single_path():
        # Single path: enumerate all subsets of path nodes
        path_nodes = []
        node = tree.root
        while node.children:
            node = next(iter(node.children.values()))
            path_nodes.append(node)
        
        for size in range(1, len(path_nodes) + 1):
            for combo in combinations(path_nodes, size):
                itemset = prefix | frozenset(n.item for n in combo)
                support = min(n.count for n in combo) / n
                yield itemset, round(support, 4)
        return
    
    # Sort items by ascending support for efficient mining
    items = sorted(
        tree.header.keys(),
        key=lambda x: tree.header[x][0]
    )
    
    for item in items:
        item_count = tree.header[item][0]
        if item_count < min_count:
            continue
        
        new_prefix = prefix | frozenset([item])
        support = item_count / n
        yield new_prefix, round(support, 4)
        
        # Build conditional pattern base
        cond_patterns = []
        for node in tree.nodes_for(item):
            path = tree.prefix_path(node)
            if path:
                cond_patterns.append((path, node.count))
        
        # Build conditional FP-Tree
        cond_tree = _build_cond_tree(cond_patterns, min_count, n)
        if cond_tree.header:
            yield from fp_growth(cond_tree, min_support, n, new_prefix)

def _build_cond_tree(cond_patterns, min_count, n):
    """Build a conditional FP-Tree from a list of (path, count) pairs."""
    # Count items across conditional patterns
    item_count = defaultdict(int)
    for path, count in cond_patterns:
        for item in path:
            item_count[item] += count
    
    # Filter by min_count
    freq = {item for item, cnt in item_count.items() if cnt >= min_count}
    
    cond_tree = FPTree()
    for path, count in cond_patterns:
        filtered = sorted(
            [item for item in path if item in freq],
            key=lambda x: (-item_count[x], x)
        )
        if filtered:
            cond_tree.insert_transaction(filtered, count)
    
    return cond_tree

# Association Rule Generation
def generate_rules(frequent_itemsets, min_confidence):
    """
    Generate association rules from frequent itemsets.
    Returns list of rule dicts sorted by lift desc.
    """
    freq_map = dict(frequent_itemsets)
    rules = []
    
    for itemset, sup in freq_map.items():
        if len(itemset) < 2:
            continue
        
        items = list(itemset)
        for size in range(1, len(items)):
            for ant_tuple in combinations(items, size):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent
                ant_sup = freq_map.get(antecedent)
                con_sup = freq_map.get(consequent)
                
                if ant_sup is None or con_sup is None:
                    continue
                
                confidence = sup / ant_sup
                if confidence < min_confidence:
                    continue
                
                lift = confidence / con_sup
                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": round(sup, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                })
    
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules

# Pretty Printing
def fmt_set(s):
    return "{" + ", ".join(sorted(s)) + "}"

def print_tree(node, indent=0, prefix="Root"):
    label = f"{prefix} [count={node.count}]" if node.item else "Root"
    print("  " + " " * indent + label)
    for child in node.children.values():
        print_tree(child, indent + 1, f"{child.item}")

def print_frequent_itemsets(freq_itemsets):
    print("\n" + "=" * 55)
    print(" FREQUENT ITEMSETS")
    print("=" * 55)
    
    by_size = defaultdict(list)
    for itemset, sup in freq_itemsets:
        by_size[len(itemset)].append((itemset, sup))
    
    for k in sorted(by_size):
        items = sorted(by_size[k], key=lambda x: -x[1])
        print(f"\n {k}-itemsets ({len(items)} found)")
        print(" " + "-" * 40)
        for itemset, sup in items:
            print(f" {fmt_set(itemset):<35} support = {sup:.2%}")

def print_rules(rules):
    print("\n" + "=" * 70)
    print(" ASSOCIATION RULES")
    print("=" * 70)
    
    if not rules:
        print(" No rules found at the given thresholds.")
        return
    
    header = f" {'Rule':<40} {'Sup':>6} {'Conf':>6} {'Lift':>6}"
    print(header)
    print(" " + "-" * 62)
    
    for r in rules:
        rule_str = f"{fmt_set(r['antecedent'])} → {fmt_set(r['consequent'])}"
        print(
            f" {rule_str:<40} "
            f"{r['support']:>5.2%} "
            f"{r['confidence']:>5.2%} "
            f"{r['lift']:>6.2f}"
        )

# Main
if __name__ == "__main__":
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
    
    print(f"\n Transactions : {len(transactions)}")
    print(f" Min support : {MIN_SUPPORT:.0%}")
    print(f" Min confidence: {MIN_CONFIDENCE:.0%}")
    
    # Step 1: Build FP-Tree
    tree, item_support, n = build_fp_tree(transactions, MIN_SUPPORT)
    
    print("\n FP-Tree structure:")
    print_tree(tree.root)
    
    print("\n Header table (item → total support count):")
    for item, (cnt, _) in sorted(tree.header.items(), key=lambda x: -x[1][0]):
        print(f" {item:<12} count={cnt} support={cnt/n:.2%}")
    
    # Step 2: Mine frequent itemsets
    freq_itemsets = list(fp_growth(tree, MIN_SUPPORT, n))
    print_frequent_itemsets(freq_itemsets)
    
    # Step 3: Generate association rules
    rules = generate_rules(freq_itemsets, MIN_CONFIDENCE)
    print_rules(rules)
    
    print(f"\n Total frequent itemsets : {len(freq_itemsets)}")
    print(f" Total association rules : {len(rules)}\n")