#!/bin/bash

set -e

run_tests=false
git_tag=""
for arg in "$@"; do
  if [[ "${arg}" == "--tests" ]]; then
    run_tests=true
  elif [[ -z "${git_tag}" ]]; then
    git_tag="${arg}"
  else
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
done
if [[ -z "${git_tag}" ]]; then
  echo "Usage: $0 VERSION [--tests]" >&2
  exit 1
fi

CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE}" ]]; then
  echo "Could not find conda. Set CONDA_EXE before running this script." >&2
  exit 1
fi
CONDA_BASE="$(${CONDA_EXE} info --base)"
export CONDA_EXE
export PATH="${CONDA_BASE}/condabin:${CONDA_BASE}/bin:${PATH}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree has changes; commit intended release contents before releasing." >&2
  git status --short >&2
  exit 1
fi
git fetch origin --tags
if ! git merge-base --is-ancestor origin/master HEAD; then
  echo "HEAD is not a fast-forward of origin/master; refusing to release." >&2
  exit 1
fi
if git rev-parse -q --verify "refs/tags/${git_tag}" >/dev/null; then
  echo "Tag ${git_tag} already exists locally; refusing to release." >&2
  exit 1
fi
if git ls-remote --exit-code --tags origin "refs/tags/${git_tag}" >/dev/null 2>&1; then
  echo "Tag ${git_tag} already exists on origin; refusing to release." >&2
  exit 1
fi

rm -f tests.log
rm -rf dist* .tox .pytest_cache build sherlockpipe-reqs *egg-info
"${CONDA_EXE}" env remove -n sherlockpipe-reqs -y 2>/dev/null || true
if [[ "${run_tests}" == true ]]; then
  "${CONDA_BASE}/bin/python" -m pip install -q tox-conda
  if "${CONDA_BASE}/bin/python" -m tox -r > tests.log; then
    tests_results=$(grep "congratulations" tests.log || true)
    if [[ -z "${tests_results}" ]]; then
      echo "Tests failed; release aborted." >&2
      exit 1
    fi
  else
    echo "Tests failed; release aborted." >&2
    exit 1
  fi
else
  echo "Skipping tests (use --tests to run them)."
fi

"${CONDA_EXE}" create --override-channels -c conda-forge -n sherlockpipe-reqs python=3.11 -y
RELEASE_PYTHON="${CONDA_BASE}/envs/sherlockpipe-reqs/bin/python"
"${RELEASE_PYTHON}" -m pip install -U pip setuptools numpy==2.1.1
sed -i '6s/.*/version = "'${git_tag}'"/' setup.py
sed -i '1s/.*/__version__ = "'${git_tag}'"/' sherlockpipe/__init__.py
"${RELEASE_PYTHON}" -m pip install -e .
git add setup.py sherlockpipe/__init__.py
git commit -m "Preparing release ${git_tag}"
git tag "${git_tag}" -m "New release"
git push origin HEAD:refs/heads/master
git push origin "refs/tags/${git_tag}"
set +e
rm -rf sherlockpipe-reqs dist* .tox .pytest_cache build *egg-info
"${CONDA_EXE}" env remove -n sherlockpipe-reqs -y 2>/dev/null || true
