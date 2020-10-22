#!/bin/bash

rm dist* -r
rm myenv -R
rm tests.log
python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install numpy
python3 -m pip install sherlockpipe
python3 -m unittest sherlockpipe.sherlock_tests 2> tests.log
tests_results=$(cat tests.log | grep -e "FAILED" -e "failed" -e "Failed" -e "error" -e "Error" -e "ERROR")
deactivate
if [[ -z "${tests_results}" ]]; then
  python3 setup.py sdist bdist_wheel
  python3 -m twine upload dist/*
else
  echo "TESTS FAILED. See tests.log"
fi
rm myenv -R
