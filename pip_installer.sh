#!/bin/bash

rm tests.log
rm dist* -r
set -e
tox -r > tests.log
tests_results=$(cat tests.log | grep "congratulations")
if ! [[ -z ${tests_results} ]]; then
  git_tag=$1
  sed -i '6s/.*/version = "'${git_tag}'"/' setup.py
  git add setup.py
  git commit -m "Preparing release ${git_tag}"
  git tag ${git_tag} -m "New release"
  git push
  python3 setup.py sdist bdist_wheel
  python3 -m twine upload dist/*
  echo "Build docker image"
  sudo docker build ./docker/ --no-cache
  docker_image_id=$(sudo docker images | awk '{print $3}' | awk 'NR==2')
  echo "Tagging docker image with tag ${git_tag}"
  sudo docker tag ${docker_image_id} sherlockpipe/sherlockpipe:latest
  sudo docker tag ${docker_image_id} sherlockpipe/sherlockpipe:${git_tag}
  echo "Push docker image with tag ${git_tag}"
  sudo docker push sherlockpipe/sherlockpipe:latest
  sudo docker push sherlockpipe/sherlockpipe:${git_tag}
else
  echo "TESTS FAILED. See tests.log"
fi
rm dist* -r
