Example of use:
===============

python v1like_extract_fromcsv.py -i ./test_imageset config_cla__v1like_a.py ./test_imageset/train5test5_split01.csv .v1like_a  --nprocessors=$(cat /proc/cpuinfo | grep processor | wc -l)

