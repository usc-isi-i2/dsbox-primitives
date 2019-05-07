#!/bin/bash

python ../dsbox/generate-primitive-json.py
cd dsbox-unit-test-datasets
git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis CI"

if [[ $TRAVIS_BRANCH == "master" ]];then
  echo "We're in master branch, will push generate json files to."
  echo "https://github.com/usc-isi-i2/dsbox-unit-test-datasets/tree/primitive_repo_master"
  git checkout -b primitive_repo_master
  rm -rf *
  mv ../output .
  git add .
  git commit -a --message "auto_generated_files"
  git remote add upstream https://${GH_TOKEN}@github.com/usc-isi-i2/dsbox-unit-test-datasets.git
  git push -f --quiet --set-upstream origin primitive_repo_master
else
  echo "We're in ${TRAVIS_BRANCH} branch, will push generate json files to."
  echo "https://github.com/usc-isi-i2/dsbox-unit-test-datasets/tree/primitive_repo_${TRAVIS_BRANCH}"
  git checkout -b primitive_repo_${TRAVIS_BRANCH}
  rm -rf *
  mv ../output .
  git add .
  git commit -a --message "auto_generated_files"
  git remote add upstream https://${GH_TOKEN}@github.com/usc-isi-i2/dsbox-unit-test-datasets.git
  git push -f --quiet --set-upstream origin primitive_repo_${TRAVIS_BRANCH}
fi
