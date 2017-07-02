#!/bin/bash

num_cores=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
[[ -z "$max_processes" ]] && max_processes=$num_cores
printf "Using up to %d parallel processes\n" "$max_processes"

# Where things are expected to be located
# Note: /dev/shm is usually stored in RAM, so if RAM is low change this to 
# a location on disk. RAM storage is preferable if available to speed up 
# random accesses and reduce disk wear
tar_path='./rawtxt'
inter_extract_path='/dev/shm/extract'
final_extract_path='./preproc'


process_file_group() {
	group_name="$1"

	# Extract the initial tar file containing the txt files
	tar_file="${group_name}_txt.tar.gz"
	tar xzf "${tar_path}/${tar_file}" -C "${inter_extract_path}/"
	if [ "$?" -ne "0" ] ; then
		printf "Error occurred when processing %s. Skipping this group (%s)\n" "$tar_file" "$group_name"
		return 1
	fi

	# Some incorrectly-tarred files have a different directory structure.
	# Thus, we must check that we actually got the expected rawtxt path
	rawtxt_path="${inter_extract_path}/${group_name}_txt"
	if [ ! -d "${rawtxt_path}" ] ; then
		possible_rawtxt_path="${inter_extract_path}/extract/${group_name}_txt"
		if [ ! -d "${possible_rawtxt_path}" ] ; then
			prinff "Failed to find rawtxt files for %s. Skipping this group (%s)\n" "$tar_file" "$group_name"
			return 1
		fi
		mv "${possible_rawtxt_path}" "${rawtxt_path}"
	fi

	# Set up somewhere to store the preprocessed files
	preproc_dir_path="${inter_extract_path}/${group_name}_preproc"
	mkdir -p "$preproc_dir_path"

	# Preprocess each txt file
	for rawtxt_file in ${rawtxt_path}/*.txt ; do
		rawtxt_filename="$(basename "${rawtxt_file}")"
		preproc_filename="$(sed 's/.txt$/.preproc/g' <<< "${rawtxt_filename}")"

		# Here's where the magic happens
		cat "${rawtxt_path}/${rawtxt_filename}" | iconv -f utf8 -t ascii//TRANSLIT /dev/stdin | tr 'A-Z' 'a-z' | sed -f paper_preproc.sed | ./stem /dev/stdin > "${preproc_dir_path}/${preproc_filename}"

		if [ "$?" -ne "0" ] ; then
			printf "Error occurred when processing %s. Continuing anyway (group %s).\n" "$rawtxt_file" "$group_name"
		fi
	done

	# Tar all the preprocessed files
	preproc_tar_filename="$(basename "${preproc_dir_path}").tar.gz"
	tar czf "${final_extract_path}/${preproc_tar_filename}" -C "$(dirname "${preproc_dir_path}")" "$(basename "${preproc_dir_path}")"

	# Wait here for legacy reasons. It allows backgrounding steps above, 
	# if one wants more fine-grained parallelization control
	wait

	# Get rid of our temporary extraction directory
	rm -r "${preproc_dir_path}" "${rawtxt_path}"
}



declare -a arg_sets
init_arg_sets() {
	for tar_file in "${tar_path}"/*.tar.gz ; do
		tar_filename="$(basename "${tar_file}")"
		group_name="$(sed 's/^\([0-9]\{1,5\}\)_txt.tar.gz/\1/g' <<< "${tar_filename}")"
		arg_sets+=("${group_name}")
	done
}
init_arg_sets

run_arg_set() {
	process_file_group $@
}

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
