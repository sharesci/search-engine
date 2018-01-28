#!/bin/bash

DIRECTORY=$1
FILES=./$DIRECTORY/*
EXTRACT=wikiextractor-url/WikiExtractor.py
OUTPUT=./$DIRECTORY/output
JSONFILES=$OUTPUT/json
COUNTER=0

if [ ! -d "$DIRECTORY" ]; then
  # Control will enter here if $DIRECTORY exists.
  echo "Directory $DIRECTORY does not exist"
  exit -1
fi

if [ ! "$EXTRACT" ]; then
  echo "Cannot find $EXTRACT"
  exit -1
fi

echo "Creating output directory: $OUTPUT"
mkdir "$OUTPUT"

if [ $? -ne 0 ] ; then
  echo "Failed to create output directory"
  exit -1
fi

echo "Creating json output directory: $JSONFILES"
mkdir "$JSONFILES"

if [ ! -d "$OUTPUT" ]; then
  echo "Failed to create output directory"
  exit -1
fi

echo "Processing files in $DIRECTORY"
echo "Processing $FILES"
for f in $FILES
do
  echo "Processing $f file..."
  python $EXTRACT --json -o $OUTPUT $f

  for json in $OUTPUT/**
  do
    if [ ! -d $f ]; then
      echo "Importing $f into mongodb collection wiki"
      mv $json "$JSONFILES/$json_$COUNTER"
      COUNTER=$((COUNTER+1))

    else
      echo "Entering directory $f..."
    fi
  done

  # take action on each file. $f store current file name
  # cat $f
done

shopt -s globstar
for f in $OUTPUT/**
do
  if [ ! -d $f ]; then
    echo "Importing $f into mongodb collection wiki"
    mongoimport --db sharesci --collection wiki --file $f
  else
    echo "Entering directory $f..."
  fi
done

