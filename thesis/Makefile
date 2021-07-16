

copy_tex:
	cp *.tex ./output_directory/


build_pdf bp2 bp3: copy_tex
	pdflatex -synctex=1 -output-directory ./output_directory thesis.tex

build_bib:
	cp thesis.bib ./output_directory
	bash -c 'cd output_directory; bibtex thesis;'

build: build_pdf build_bib bp2 bp3
	echo "built"

clean:
	rm -f output_directory/thesis.bbl
	rm -f output_directory/thesis.toc
	rm -f output_directory/thesis.lot
	rm -f output_directory/thesis.synctex.gz
	rm -f output_directory/thesis.blg
	rm -f output_directory/thesis.lof
	rm -f output_directory/thesis.bib
	rm -f output_directory/thesis.aux
	rm -f output_directory/thesis.log
	rm -f output_directory/thesis.out

cleaner: clean
	rm -f output_directory/thesis.pdf

all:
	build
	

# format_bib:
# 	biber --tool --output_align --output_indent=2 --output_fieldcase=lower thesis.bib
	
