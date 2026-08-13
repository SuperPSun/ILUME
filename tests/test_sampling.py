from __future__ import annotations

from stage1.sampler import allocate_role_quotas, coverage_epoch_plan


def test_formal_sampling_is_exact_45_45_10_with_full_coverage() -> None:
    assert allocate_role_quotas(1000) == (450, 450, 100)
    plan = coverage_epoch_plan((24908, 27907, 56532), 256, 1)
    assert plan.role_quotas == (254477, 254477, 56550)
    assert all(
        quota >= count
        for quota, count in zip(
            plan.role_quotas, (24908, 27907, 56532), strict=True
        )
    )
