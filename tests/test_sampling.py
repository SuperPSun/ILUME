from __future__ import annotations

from stage1.train import _EpochSampler


SHARDS = ((0, 4), (4, 3), (7, 4))


def _sampler(*, epoch=0, cursor=0, rank=0, world_size=1):
    return _EpochSampler(
        SHARDS,
        size=11,
        seed=17,
        epoch=epoch,
        cursor=cursor,
        rank=rank,
        world_size=world_size,
    )


def test_epoch_sampler_is_deterministic_unique_and_changes_by_epoch() -> None:
    first = list(_sampler())
    assert first == list(_sampler())
    assert len(first) == len(set(first)) == 11
    assert set(first) == set(range(11))
    assert first != list(_sampler(epoch=1))


def test_epoch_sampler_cursor_replays_exact_tail() -> None:
    complete = list(_sampler())
    assert list(_sampler(cursor=5)) == complete[5:]
    assert len(_sampler(cursor=5)) == len(complete) - 5


def test_epoch_sampler_ddp_partitions_and_only_pads_prefix() -> None:
    partitions = [list(_sampler(rank=rank, world_size=4)) for rank in range(4)]
    assert {len(values) for values in partitions} == {3}
    flattened = [value for values in partitions for value in values]
    counts = {value: flattened.count(value) for value in set(flattened)}
    assert set(counts) == set(range(11))
    assert sum(count - 1 for count in counts.values()) == 1

    global_order = list(_sampler())
    duplicate = next(value for value, count in counts.items() if count == 2)
    assert duplicate == global_order[0]


def test_epoch_sampler_keeps_each_shard_contiguous_in_global_order() -> None:
    selected = list(_sampler())
    shard_by_index = {
        index: shard
        for shard, (start, count) in enumerate(SHARDS)
        for index in range(start, start + count)
    }
    shard_sequence = [shard_by_index[index] for index in selected]
    changes = sum(
        left != right
        for left, right in zip(shard_sequence, shard_sequence[1:])
    )
    assert changes == len(SHARDS) - 1
