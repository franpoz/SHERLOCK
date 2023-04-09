#!/bin/bash

rm tests.log
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -r sherlockpipe-reqs
rm -R *egg-info
set -e

git_tag=$1
echo "GIT TAG IS " ${git_tag}
echo "Run regression tests"
tox -r -e py38-local,py39-local > tests.log
tests_results=$(cat tests.log | grep "congratulations")
if ! [[ -z ${tests_results} ]]; then
  echo "Run all tests"
  set +e
  rm tests.log
  rm -r .tox
  rm -r .pytest_cache
  rm -r build
  rm -r sherlockpipe-reqs
  rm -R *egg-info
  set -e
  tox -r -e py38-gha,py39-gha > tests.log
else
  echo "TESTS FAILED. See tests.log"
  set +e
  rm -r .tox
  rm -r .pytest_cache
  rm -r build
  rm -r sherlockpipe-reqs
  rm -R *egg-info
  set -e
  exit 1
fi
tests_results=$(cat tests.log | grep "congratulations")
if ! [[ -z ${tests_results} ]]; then
  set +e
  rm tests.log
  rm -r .tox
  rm -r .pytest_cache
  rm -r build
  rm -r sherlockpipe-reqs
  rm -R *egg-info
  set -e
  python3.8 -m venv sherlockpipe-reqs
  source sherlockpipe-reqs/bin/activate
  python3.8 -m pip install pip -U
  python3.8 -m pip install numpy==1.22.4
  sed -i '6s/.*/version = "'${git_tag}'"/' setup.py
  sed -i '1s/.*/__version__ = "'${git_tag}'"/' sherlockpipe/__init__.py
  python3.8 -m pip install -e .
  python3.8 -m pip list --format=freeze > requirements.txt
  deactivate
  git add requirements.txt
  git add setup.py
  git add sherlockpipe/__init__.py
  git commit -m "Preparing release ${git_tag}"
  git tag ${git_tag} -m "New release"
  git push
  git push --tags
#  python3 setup.py sdist bdist_wheel
#  python3 -m twine upload dist/*
#  echo "Build docker image"
#  sudo docker build ./docker/ --no-cache
#  docker_image_id=$(sudo docker images | awk '{print $3}' | awk 'NR==2')
#  echo "Tagging docker image with tag ${git_tag}"
#  sudo docker tag ${docker_image_id} sherlockpipe/sherlockpipe:latest
#  sudo docker tag ${docker_image_id} sherlockpipe/sherlockpipe:${git_tag}
#  echo "Push docker image with tag ${git_tag}"
#  sudo docker push sherlockpipe/sherlockpipe:latest
#  sudo docker push sherlockpipe/sherlockpipe:${git_tag}
#  sudo docker images prune -all
#  rm tests.log
else
  echo "TESTS FAILED. See tests.log"
fi
set +e
rm -R sherlockpipe-reqs
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -R *egg-info
