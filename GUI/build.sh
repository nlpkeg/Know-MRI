# !/bin/bash

cd front
npm run build
cd ..

find templates/ -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
find static/ -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

mv front/dist/index.html templates/
mv front/dist/* static/

