import tskit
import pyslim
import numpy as np

ts = tskit.load("../../temp/founders.trees")


def founder_stats(ts, root_time):
    """
    Returns arrays with (a) the total amount of genome on which
    each founder has left genetic material, and (b) the total amount
    of genetic material in the final population. This is calculated
    as a proportion of the genome, averaged across chromosomes,
    so a diploid founder who has descendants carrying one chromosome
    but not the other will have "has_offspring" of 0.5.
    """

    founder_indivs = pyslim.individuals_alive_at(ts, root_time)
    print("Founder individuals:", founder_indivs)
    modern_indivs = pyslim.individuals_alive_at(ts, 0.0)
    print("Modern individuals:", modern_indivs)
    modern_nodes = np.array([n for i in modern_indivs for n in ts.individual(i).nodes])
    print("Modern nodes:", modern_nodes)

    # Only map FOUNDER nodes to founder indices
    node_indexes = np.full((ts.num_nodes,), -1, dtype='int')
    for k, i in enumerate(founder_indivs):
        ind = ts.individual(i)
        print(f"Founder {k}: Individual {i}, nodes {ind.nodes}")
        for n in ind.nodes:
            node_indexes[n] = k

    # Remove this loop - it was overwriting the founder mapping!
    # for k, i in enumerate(modern_indivs):
    #     ind = ts.individual(i)
    #     print(ind)
    #     for n in ind.nodes:
    #         node_indexes[n] = k

    print("\nAnalyzing trees:")
    has_offspring = np.zeros(len(founder_indivs))
    total_offspring = np.zeros(len(founder_indivs))

    for tree_idx, t in enumerate(ts.trees(tracked_samples=modern_nodes)):
        print(f"\nTree {tree_idx}: {t.interval}")
        print(f"  Roots: {list(t.roots)}")

        for r in t.roots:
            k = node_indexes[r]
            if k < 0:
                print(f"  Warning: Root {r} not mapped to any founder")
                continue

            f = t.num_tracked_samples(r)
            print(f"  Root {r} -> Founder {k}, tracked samples: {f}")

            has_offspring[k] += t.span * (f > 0)
            total_offspring[k] += t.span * f

    has_offspring /= ts.sequence_length  # Remove the factor of 2 since we're counting genome segments
    total_offspring /= ts.sequence_length

    return has_offspring, total_offspring


# Also let's add a function to trace ancestry segments
def trace_ancestry_segments(ts):
    """
    Trace which founder each segment comes from for each modern individual
    """
    root_time = np.max(ts.tables.nodes.time)
    founder_indivs = pyslim.individuals_alive_at(ts, root_time)
    modern_indivs = pyslim.individuals_alive_at(ts, 0.0)

    # Map founder nodes to founder indices
    node_to_founder = {}
    for founder_idx, ind_id in enumerate(founder_indivs):
        ind = ts.individual(ind_id)
        for node in ind.nodes:
            node_to_founder[node] = founder_idx

    print("\n" + "=" * 60)
    print("ANCESTRY TRACING")
    print("=" * 60)

    for modern_idx, ind_id in enumerate(modern_indivs):
        ind = ts.individual(ind_id)
        print(f"\nModern Individual {modern_idx} (ID: {ind_id}):")

        for chrom_idx, sample_node in enumerate(ind.nodes):
            print(f"  Chromosome {chrom_idx} (node {sample_node}):")

            segments = []
            current_founder = None
            segment_start = 0

            for tree in ts.trees():
                # Find which founder this sample traces to in this tree
                founder = None

                # Walk up the tree to find a founder node
                current = sample_node
                path = []
                while current != tskit.NULL:
                    path.append(current)
                    if current in node_to_founder:
                        founder = node_to_founder[current]
                        break
                    # Move to parent
                    parent = tree.parent(current)
                    current = parent

                # If founder changed, record the previous segment
                if founder != current_founder:
                    if current_founder is not None:
                        segments.append({
                            'start': segment_start,
                            'end': tree.interval.left,
                            'founder': current_founder,
                            'length': tree.interval.left - segment_start
                        })
                    current_founder = founder
                    segment_start = tree.interval.left

            # Add the final segment
            if current_founder is not None:
                segments.append({
                    'start': segment_start,
                    'end': ts.sequence_length,
                    'founder': current_founder,
                    'length': ts.sequence_length - segment_start
                })

            # Print segments for this chromosome
            for i, seg in enumerate(segments):
                crossover_marker = " <- CROSSOVER" if i > 0 else ""
                print(f"    {seg['start']:>8.0f} - {seg['end']:>8.0f} "
                      f"(len: {seg['length']:>8.0f}) -> Founder {seg['founder']}{crossover_marker}")


root_time = np.max(ts.tables.nodes.time)
print(f"Root time: {root_time}")

has_offspring, total_offspring = founder_stats(ts, root_time)
print(f"\nFounder stats:")
print(f"Has offspring: {has_offspring}")
print(f"Total offspring: {total_offspring}")

# Now trace the actual ancestry segments
trace_ancestry_segments(ts)