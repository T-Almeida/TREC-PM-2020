#!/bin/bash

echo "Install python the requirements"
python install -r requirements.txt

echo "Manually Install mmnrm python library"
cd _other_dependencies/mmnrm/
rm -r ./dist
python setup.py sdist
pip install ./dist/mmnrm-0.0.2.tar.gz

echo "Manually Install nir python library"
cd ../nir/
rm -r ./dist
python setup.py sdist
pip install ./dist/nir-0.0.1.tar.gz

cd ../../

