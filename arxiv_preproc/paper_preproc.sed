# First, loop until we've read the whole file into the pattern space
:addToBuffer
$b doReplace;
N;
b addToBuffer;


# Real replacements start here
:doReplace

# Join words broken over lines
s/-[ ]\?\n//g;

# Normalize the layout. Put everything on one line
s/\n/ /g;

# Remove quotes
#s/["\x201C\x201D]\|\s'\|'\s/ /g

# Words should be separated by exactly one space
s/\s\+/ /g;

# Remove punctuation
#s/\([A-Za-z0-9]\+\)[.,!?] /\1 /g

# A trick: put spaces at the start and end of the file so we can say a
# word is anything surrounded by spaces
s/^/ /g
s/$/ /g

# Now that we know what a "word" is, we can get rid of words that we
# don't want.
# First, get rid of numeric references
s/ \?\[[0-9]\+\]//g

# Now get rid of any words containing any non-word characters. We use
# an alternative definition from the regex definition of "word"
# characters, and ours includes numbers and hyphens in addition to the
# usual alphabetics and underscores
#s/ [^ ]*[^ A-Za-z0-9_-][^ ]* / /g

# Finally, nuke any remaining special characters anywhere in the 
# pattern space
s/[^ A-Za-z0-9_-.!?:;@$%^*&<>{}]//g

# Get rid of any numbers (note: deleting numerics
# character-by-character instead of deleting "words containing
# numbers" because sometimes real words get concatenated with numbers)
s/[0-9]\+//g

# Get rid of any non-wordlike characters
s/[^\sA-Za-z_-]\+/ /g

# Split words onto their own line each (useful for others in the 
# preprocessing pipeline, like the stemmer)
s/\(\b\w[A-Za-z_-]\+\b\)/\n\1\n/g
s/\n\s\+/\n/g
s/\s\+\n/\n/g

