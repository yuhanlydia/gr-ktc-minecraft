from gr_ktc.retry_loop import VerifiedRollout, run_verified_retry_loop


def test_retry_loop_merges_mixed_group_then_stops_on_success():
    calls = []

    def rollout(memory, cycle, index):
        calls.append((memory, cycle, index))
        score = 1.0 if cycle == 1 and index == 0 else float(index == 0 and cycle == 0)
        # Avoid immediate success in cycle 0.
        if cycle == 0:
            score *= 0.5
        return VerifiedRollout(f"t-{cycle}-{index}", score, True)

    def merge(trajectories, advantages):
        return "merged"

    result = run_verified_retry_loop(rollout, merge, success_score=1.0)
    assert result.stopped_reason == "verified_success"
    assert len(result.cycles) == 2
    assert calls[4][0] == "merged"


def test_all_failure_group_does_not_invent_memory():
    def rollout(memory, cycle, index):
        return VerifiedRollout(index, 0.0, True)

    result = run_verified_retry_loop(
        rollout, lambda trajectories, advantages: "should-not-run", max_cycles=1
    )
    assert result.cycles[0].memory is None
    assert result.stopped_reason == "budget_exhausted"

