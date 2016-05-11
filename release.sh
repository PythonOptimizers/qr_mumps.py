#!/bin/sh
#
# Runs before git flow release finish
tobranch="master"

git branch -D $tobranch
git push origin --delete $tobranch
git checkout -b $tobranch

git rm --cached \*.cpy
git rm --cached \*.cpx
git rm --cached \*.cpd
git rm --cached generate_code.py
git rm --cached release.sh

mv .gitignore /tmp/.gitignore-qr_mumps

cp config/site.template.cython.cfg site.cfg

python generate_code.py -c
python generate_code.py
python setup.py install

git add \*.c
git rm --cached -r build
git rm --cached -r config

cp config/site.template.cfg .
cp config/.gitignore .
cp config/.travis.yml .

git add tests/\*.py
git add setup.py
git add --all

git commit -m "c files from last commit in develop"
git push --set-upstream origin $tobranch
git checkout develop
