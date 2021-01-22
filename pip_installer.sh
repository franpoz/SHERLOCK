#!/bin/bash

rm dist* -r
rm myenv -R
rm tests.log
python3.8 -m venv myenv
source myenv/bin/activate
export LLVM_CONFIG=/usr/bin/llvm-config-10
python3.8 -m pip install wheel
python3.8 -m pip install numpy
python3.8 -m pip install cython
python3.8 -m pip install pandas
python3.8 -m pip install lightkurve
python3.8 -m pip install transitleastsquares
python3.8 -m pip install requests
python3.8 -m pip install wotan
python3.8 -m pip install matplotlib
python3.8 -m pip install pyyaml
python3.8 -m pip install allesfitter
python3.8 -m pip install seaborn
python3.8 -m pip install bokeh
python3.8 -m pip install astroplan
python3.8 -m pip install astroquery
python3.8 -m pip install sklearn
python3.8 -m pip install scipy
python3.8 -m pip install tess-point
python3.8 -m pip install reproject==0.4
python3.8 -m pip install reportlab
python3.8 -m pip install astropy
python3.8 -m pip install mock > 2.0.0
python3.8 -m pip install photutils>=0.7
python3.8 -m pip install tqdm
python3.8 -m pip install setuptools>=41.0.0
python3.8 -m pip install torch
python3.8 -m pip install beautifulsoup4>=4.6.0
python3.8 -m pip install tess-point>=0.3.6
python3.8 -m pip install nose
python3.8 -m pip install triceratops==0.2.2
python3.8 -m unittest sherlockpipe/sherlock_tests.py 2> tests.log
tests_results=$(cat tests.log | grep -e "FAILED" -e "failed" -e "Failed" -e "error" -e "Error" -e "ERROR")
deactivate
if [[ -z "${tests_results}" ]]; then
  python3 setup.py sdist bdist_wheel
  python3 -m twine upload dist/*
  echo "Build docker image"
  sudo docker build ./docker/ --no-cache
  git_tag=$(git tag -l --sort -version:refname | head -n 1)
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
rm myenv -R
rm dist* -r
