#!/bin/bash

cd tools/webui
rm -rf node_modules
npm install
npm run format
npm run lint
npm run check
npm run build
