#!/bin/bash

num_cores=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
[[ -z "$max_processes" ]] && max_processes=$num_cores
printf "Using up to %d parallel processes\n" "$max_processes"

declare -a arg_sets
init_arg_sets() {
	for file in ../../arxiv/preproc_split_subset/d* ; do
		arg_sets+=("$file")
	done
}
init_arg_sets

run_arg_set() {
	python3 bigram_indexer.py -d "$1" --new-docs -m ./results2.json
}

ulimit -v 10000000

# Run commands
cur_num_processes=0
for arg_set in "${arg_sets[@]}"
do

	printf "Starting arg set: %s\n" "$arg_set"

	(
		run_arg_set "$arg_set"
	) &

	(( cur_num_processes++ ))
	if (( max_processes <= cur_num_processes ))
	then
		wait -n
		(( cur_num_processes-- ))
	fi

done

wait

printf "All done.\n"
