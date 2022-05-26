BOOK_DIR := ../OPSI_ML_workshop/
BUILD_DIR := _build/html

book:

	jupyter-book build ${BOOK_DIR}

from_scratch:

	## Builds without cached files
	jupyter-book build -all ${BOOK_DIR}

public:

	ghp-import -n -p -f ${BOOK_DIR}${BUILD_DIR}

# clean:
#         rm -f ${TEX_DIR}/*.{ps,log,aux,out,dvi,bbl,blg,fls,log,aux,fdb_latexmk,synctex.gz,toc,xdv}

open:

		open ${BOOK_DIR}${BUILD_DIR}/index.html
