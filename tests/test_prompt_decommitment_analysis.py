from scripts.analyze_prompt_decommitment import paired_rows, summarize_mode


def _row(scene, mode, msr, success, hop):
    return {"scene_id": scene, "mode": mode, "msr": msr, "task_success": success, "hop_count": hop}


def test_paired_rows_pairs_same_scene_only():
    rows = [
        _row("a", "reflection", 0.25, False, 2),
        _row("a", "decommit", 0.75, True, 2),
        _row("b", "reflection", 0.50, False, 3),
    ]
    pairs = paired_rows(rows, "decommit", "reflection")
    assert len(pairs) == 1
    assert pairs[0][0]["scene_id"] == "a"
    assert pairs[0][1]["scene_id"] == "a"


def test_summarize_mode_reports_tsr_and_mean_msr():
    rows = [
        _row("a", "decommit", 1.0, True, 2),
        _row("b", "decommit", 0.5, False, 3),
    ]
    summary = summarize_mode(rows, "decommit")
    assert summary["n"] == 2
    assert summary["tsr"] == 0.5
    assert summary["mean_msr"] == 0.75


def test_stratified_scene_selection_uses_metadata_hop_counts(tmp_path):
    from scripts.run_mineexplorer_prompt_decommitment import _scenario_entries

    for index, hops in enumerate([1, 1, 2, 2, 3, 3, 4, 4], 1):
        meta = tmp_path / f"{index:04d}" / "multi-agent" / "metadata.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(__import__("json").dumps({
            "task_text": "x",
            "milestones": [{"milestone_id": str(i), "rules": []} for i in range(hops)],
        }))

    entries = _scenario_entries(tmp_path, limit=None, per_hop=1, hop_counts=(1, 2, 3, 4))

    assert len(entries) == 4
    assert [entry[2] for entry in entries] == [1, 2, 3, 4]
