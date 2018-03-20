#!/bin/bash


dataset=1

for seed in $(seq 1 5); do
	for points in 1; do
		for tasks in 75; do
			for dims in 50; do
				for method in 0 1 2 3; do

					case $method in
						0|2|3)

						case $method in
							0)
							RAM=2.3G
							TIME=2:30:00
							;;
							2)
							RAM=2.2G
							TIME=2:30:00
							;;
							3)
							RAM=2.4G
							TIME=1:10:00
						esac

		                for lambda in $(seq 0 24); do

		                	qsub -N ss.$seed.$method.$points.$tasks.$lambda -l tmem=$RAM -l h_vmem=$RAM -l h_rt=$TIME schools_python.sh $seed $points $tasks $dims $method $dataset $lambda
						done;
						sleep 0.1;
						;;
						1)

						RAM=2.3G
						TIME=2:30:00

		                for cvalue in 0.1 1000 100000 1000000000000; do
							qsub -N ss.$seed.$method.$points.$tasks.$cvalue -l tmem=$RAM -l h_vmem=$RAM -l h_rt=$TIME schools_python.sh $seed $points $tasks $dims $method $dataset $cvalue
							sleep 0.1;
						done;
					esac

				done;
			done;
		done;
	done;
done
