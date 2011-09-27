(TeX-add-style-hook "phi4"
 (lambda ()
    (LaTeX-add-bibliographies)
    (LaTeX-add-labels
     "sec:level1"
     "fig.1"
     "tab:tableA"
     "tab:tableB"
     "tab:tableC"
     "tab:tableD"
     "tab:Deviates"
     "fig:pca"
     "tab:tableOI"
     "tab:tableE"
     "fig.hier")
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

