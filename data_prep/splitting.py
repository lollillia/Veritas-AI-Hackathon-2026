"""Stratified, group-aware train/dev/test assignment."""
import random

SPLITS = ("train", "dev", "test")


def assign_splits(rows: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 0) -> list[str]:
    """Assign each row to "train", "dev" or "test".

    Rows sharing a group_id always get the same split. Within each (source, label) stratum
    the groups are shuffled with the seed and handed out greedily to whichever split is
    furthest below its target share of that stratum's images. A group that spans strata is
    handled with its first member's stratum. Returns one split per input row, in order.
    """
    groups: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(i)

    strata: dict[tuple, list[int]] = {}
    for gid, members in groups.items():
        first = rows[members[0]]
        strata.setdefault((first["source"], first["label"]), []).append(gid)

    out = [""] * len(rows)
    for key in sorted(strata):
        gids = strata[key]
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        targets = [total * r for r in ratios]
        have = [0, 0, 0]
        for gid in gids:
            deficits = [t - h for t, h in zip(targets, have)]
            k = max(range(3), key=lambda j: (deficits[j], -j))   # ties go to the earlier split
            for i in groups[gid]:
                out[i] = SPLITS[k]
            have[k] += len(groups[gid])
    return out
