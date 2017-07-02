#!/bin/bash

num_cores=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
[[ -z "$max_processes" ]] && max_processes=$num_cores
printf "Using up to %d parallel processes\n" "$max_processes"

tar_path='../pdf'
inter_extract_path='../extract'
final_extract_path='../extract'

groupinfo_dir='./groupinfo'
grouplist_file='./grouplist.txt'

process_file_group() {
	group_name="$1"
	shift
	for tar_file in $@; do
		echo $tar_file
#		echo rsync 'bigdata1:/bdata/bigdata1/sharesci/arxiv/pdf/'"${tar_file}" "${tar_path}"
		tar xvf "${tar_path}/${tar_file}" -C "${inter_extract_path}" > /dev/null
		if [ "$?" -ne "0" ] ; then
			printf "Error occurred when processing %s. Skipping this group (%s)\n" "$tar_file" "$group_name"
			return 1
		fi
#		rm "${tar_path}/${tar_file}"
	done
	pdf_path="${inter_extract_path}/${group_name}"
	txt_dir_path="${inter_extract_path}/${group_name}_txt"
	mkdir -p "$txt_dir_path"
	for pdf_file in ${pdf_path}/*.pdf ; do
		pdf_filename="$(basename "${pdf_file}")"
		txt_filename="$(sed 's/.pdf$/.txt/g' <<< "${pdf_filename}")"
		pdftotext "${pdf_path}/${pdf_filename}" "${txt_dir_path}/${txt_filename}"
		if [ "$?" -ne "0" ] ; then
			printf "Error occurred when processing %s. Continuing anyway (group %s).\n" "$pdf_file" "$group_name"
		fi
	done
	txt_tar_filename="$(basename "${txt_dir_path}").tar.gz"
	tar cvzf "${final_extract_path}/${txt_tar_filename}" -C "$(dirname "${txt_dir_path}")" "$(basename "${txt_dir_path}")" > /dev/null;
	rm -r "${txt_dir_path}" "${pdf_path}"
#	echo rsync "${final_extract_path}/$(basename "${txt_tar_file_name}")" 'bigdata1:/bdata/bigdata1/sharesci/arxiv/extract/'
}


declare -A tar_groups
init_tar_groups() {
	for file in "${groupinfo_dir}"/*.txt ; do
		if [ ! -f "$file" ] ; then
			continue
		fi
		if [ -z "$(egrep '[0-9]{1,5}.txt$' <<< "$file")" ] ; then
			continue
		fi
		groupname="$(sed -r -e 's/[^0-9]*([0-9]{1,5}).txt$/\1/g' <<< "$file")"
		mapfile -t arr < "$file"
#		echo "$groupname"
#		echo "${arr[@]}"
#		sleep 0.1
		tar_groups["$groupname"]="${arr[@]}"
	done
}
init_tar_groups

declare -a arg_sets
init_arg_sets() {
	mapfile -t needed_groups < "${grouplist_file}"
	for group in "${needed_groups[@]}" ; do
		arg_sets+=("${group} ${tar_groups[$group]}")
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
