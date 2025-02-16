#! /bin/bash
set -e

# light theme
rm -rf tmp
python3 generate_gif.py $1 light
cd tmp
inkscape --export-type=png *.svg
convert *.png "$1"_light.gif
mv "$1"_light.gif ../
cd ..

# dark theme
rm -rf tmp
python3 generate_gif.py $1 dark
cd tmp
inkscape --export-type=png *.svg
convert *.png "$1"_dark.gif
mv "$1"_dark.gif ../
cd ..

rm -rf tmp
