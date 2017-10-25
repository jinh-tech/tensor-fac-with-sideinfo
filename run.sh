#!/bin/bash

# number	algo	dataset		rank	test 	iter

# 1. 		hptfsi duke_awv 	k=10 	False 	101

if [ $1 -eq 1 ]
	then
	echo python code/hptfsi.py -d=data/duke_awv/duke_awv.npz -o=temp_results/ -k=10 -v -n=101
	python code/hptfsi.py -d=data/duke_awv/duke_awv.npz -o=temp_results/ -k=10 -v -n=101
fi

