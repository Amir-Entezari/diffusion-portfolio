from pathlib import Path


def test_unvalidated_proposal_components_are_not_active_modules():
    package = Path(__file__).parents[2] / "src" / "diffusion_portfolio"
    active_files = {p.name.lower() for p in package.rglob("*.py")}
    assert "sliced_gw.py" not in active_files
    assert "sgw_cross_attention.py" not in active_files
    assert "jump_process.py" not in active_files
    assert "bsde_controller.py" not in active_files
    assert "ttsa_trainer.py" not in active_files
