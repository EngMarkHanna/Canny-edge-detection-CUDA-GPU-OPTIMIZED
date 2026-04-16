#!/usr/bin/env bash
image_path="../pictures/images_4k/0a866770efb32f7d.png"

python test_canny.py "$image_path" 50 150 8
./canny_all.x "$image_path" 50 150 8