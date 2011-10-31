(TeX-add-style-hook "musimetrics"
 (lambda ()
    (LaTeX-add-bibliographies)
    (LaTeX-add-labels
     "sec:level1"
     "sec:results"
     "tab:table0"
     "tab:tableA"
     "tab:tableB"
     "tab:tableC"
     "tab:tableD"
     "tab:Deviates"
     "fig:pca"
     "tab:tableOI"
     "tab:tableE"
     "tab:varphi"
     "fig:phipca"
     "tab:tablephiOI"
     "tab:tablephiE"
     "fig:comparingdialectics"
     "fig:dendrogram")
    (TeX-run-style-hooks
     "color"
     "multirow"
     "bm"
     "dcolumn"
     "grffile"
     "graphicx"
     "latex2e"
     "revtex4-110"
     "revtex4-1"
     "aip"
     "jmp"
     "amsmath"
     "amssymb"
     "reprint")))

