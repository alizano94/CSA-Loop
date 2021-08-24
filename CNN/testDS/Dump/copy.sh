#!/bin/bash

for i in {1..16}
do
	echo $i
	shuf -zn1 -e ./0/*.png | xargs -0 cp -vt ../test/0/
done

