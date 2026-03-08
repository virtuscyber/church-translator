from __future__ import annotations


def test_launcher_scripts_exist_and_contain_expected_commands(project_root):
    start_sh = project_root / "start.sh"
    start_command = project_root / "start.command"
    start_bat = project_root / "start.bat"

    assert start_sh.exists()
    assert start_command.exists()
    assert start_bat.exists()

    sh_text = start_sh.read_text(encoding="utf-8")
    command_text = start_command.read_text(encoding="utf-8")
    bat_text = start_bat.read_text(encoding="utf-8")

    assert "#!/bin/bash" in sh_text
    assert 'cd "$(dirname "$0")"' in sh_text
    assert "python3 run.py" in sh_text

    assert "#!/bin/bash" in command_text
    assert 'cd "$(dirname "$0")"' in command_text
    assert "python3 run.py" in command_text

    assert "@echo off" in bat_text
    assert 'cd /d "%~dp0"' in bat_text
    assert "python run.py" in bat_text
