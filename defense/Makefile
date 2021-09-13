
.DEFAULT_GOAL := all

bib:
	bibtex output_directory/defense.aux

notes:
	pdflatex --output-directory output_directory notes.tex

all:
	pdflatex --output-directory output_directory defense.tex
