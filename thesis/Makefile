
build_pdf bp2 bp3:
	pdflatex -synctex=1 -output-directory ./output_directory thesis.tex

build_bib:
	cp thesis.bib ./output_directory
	bash -c 'cd output_directory; bibtex thesis;'

build: build_pdf build_bib bp2 bp3
	echo "built"

all:
	build
	

# format_bib:
# 	biber --tool --output_align --output_indent=2 --output_fieldcase=lower thesis.bib


clean:
	rm -rf output_directory/*

cleaner:
	echo "Implement me"
	
