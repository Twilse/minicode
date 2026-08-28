from __future__ import annotations

from pathlib import Path

import pytest

from minicoder.domain.errors import ConfigurationError
from minicoder.tools.safety import (
    INVALID_PATH,
    PATH_OUTSIDE_WORKSPACE,
    WorkspacePathError,
    WorkspacePathPolicy,
)


def test_policy_resolves_and_displays_a_normal_relative_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "src"
    source.mkdir()
    target = source / "main.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    policy = WorkspacePathPolicy(workspace)

    resolved = policy.resolve("src/../src/main.py")

    assert resolved == target
    assert policy.display(resolved) == "src/main.py"


def test_policy_allows_the_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspacePathPolicy(workspace)

    assert policy.resolve(".") == workspace
    assert policy.display(workspace) == "."


@pytest.mark.parametrize("raw_path", ["../outside.txt", "nested/../../outside.txt"])
def test_policy_rejects_parent_traversal(tmp_path: Path, raw_path: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(WorkspacePathError) as captured:
        policy.resolve(raw_path)

    assert captured.value.error_code == PATH_OUTSIDE_WORKSPACE


def test_policy_rejects_an_absolute_path_even_when_it_is_inside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(WorkspacePathError) as captured:
        policy.resolve(str(workspace / "main.py"))

    assert captured.value.error_code == PATH_OUTSIDE_WORKSPACE


def test_policy_rejects_a_symlink_that_escapes_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "secret-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(WorkspacePathError) as captured:
        policy.resolve("secret-link.txt")

    assert captured.value.error_code == PATH_OUTSIDE_WORKSPACE


def test_policy_rejects_creation_beneath_an_external_directory_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "linked-directory"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(WorkspacePathError) as captured:
        policy.resolve("linked-directory/new.py")

    assert captured.value.error_code == PATH_OUTSIDE_WORKSPACE


@pytest.mark.parametrize("raw_path", ["", "bad\x00path"])
def test_policy_rejects_invalid_path_text(tmp_path: Path, raw_path: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(WorkspacePathError) as captured:
        policy.resolve(raw_path)

    assert captured.value.error_code == INVALID_PATH


def test_policy_rejects_a_missing_or_non_directory_workspace(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        WorkspacePathPolicy(tmp_path / "missing")
    with pytest.raises(ConfigurationError):
        WorkspacePathPolicy(file_path)
