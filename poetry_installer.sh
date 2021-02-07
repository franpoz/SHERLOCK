#!/bin/bash

rm dist* -r
tox -r > tests.log
tests_results=$(cat tests.log | grep "congratulations")
if ! [[ -z ${tests_results} ]]; then
  git_tag=$1
  sed -i '3s/.*/version = "'${git_tag}'"/' pyproject.toml
  git add pyproject.toml
  git commit -m "Preparing release ${git_tag}"
  git tag ${git_tag} -m "New release"
  git push
  poetry build
  poetry publish
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
