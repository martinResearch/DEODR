# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
# Copyright © 2016 Martin de la Gorce <martin[dot]delagorce[hat]gmail[dot]com>

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------


# This script will parse you readme.md file and create images for each equation found as
# [latex:your_equation](your_image_file)
#
# we recommand using svg image files as they give nice vectorial images without pixel
# aliasing and they are in a text format which is good for versioning with git or
# mercurial # has the svg text remains unchanged for unchanged euation and thus it
# avoids pushing again and again the same # images on the server of the aleready
# compile equations each time a new equation is created Martin de La Gorce April 2016
# up to date version of that script can found on
# https://github.com/martinResearch/markdownLatex

"""Script to render latex figure for each equation found in readme.md"""


import os
import re
import shutil
import sys
import tempfile


for arg in sys.argv:
    print(arg)

dirpath = tempfile.mkdtemp()
# ... do stuff with dirpath
print(f"temporary directory for latex compilation = {dirpath}")
if len(sys.argv) == 1:
    texfile = "./readme.md"
elif len(sys.argv) == 2:
    texfile = sys.argv[1]
else:
    raise Exception("wrong number of arguments")


def formula_as_file(
    formula: str, file: str, negate: bool = False, header: str = ""
) -> None:
    laxtex_tmp_file = os.path.join(dirpath, "tmp_equation.tex")
    pdf_tmp_file = os.path.join(dirpath, "tmp_equation.pdf")
    with open(laxtex_tmp_file, "w") as latexfile:
        latexfile.write("\\documentclass[preview]{standalone}")
        # latexfile.write('\\input{header.tex}')
        latexfile.write("\\usepackage{wasysym}")
        latexfile.write("\\usepackage{amssymb}")
        latexfile.write("\n\\begin{document}")
        latexfile.write(f" {formula}")
        latexfile.write("\n\\end{document}  ")
    os.system('pdflatex -output-directory="%s"  %s' % (dirpath, laxtex_tmp_file))
    if file.startswith("https://rawgithub.com") or file.startswith(
        "https://raw.githack.com"
    ):
        file = "./" + re.findall(r"""/master/(.*)""", file)[0]
    if file.endswith("svg"):
        os.system(f"pdf2svg {pdf_tmp_file} {file}")
    elif file.endswith("pdf"):
        shutil.copyfile(pdf_tmp_file, file)
    else:
        os.system(f"convert -density 100  {pdf_tmp_file} -quality 90  {file}")


raw = open(texfile)
filecontent = raw.read()

latex_equations = re.findall(r"""[^\t]!\[latex:(.*?)\]\((.*?)\)""", filecontent)
print("found %d equations " % len(latex_equations))
listname = set()
for eqn in latex_equations:
    if eqn[1] in listname:
        raise Exception(f"equation image file {eqn[1]} already used")

    listname.add(eqn[1])
    print(f"creating {eqn[1]}")
    formula_as_file(eqn[0], eqn[1])
    print("done")

# shutil.rmtree(dirpath)
print("DONE")
