# TensorFlow and Keras companion

This 21-page guide follows the existing Week 1 notebook 02, including its editable data-count, architecture and optimizer experiments.

From this directory, install NumPy, SciPy, Matplotlib, Pillow, ReportLab and PyMuPDF, then run:

    python build_figures.py
    python build_guide.py

The PDF is written to lectures/week01_02_tensorflow_keras.pdf.
All eight diagrams are generated from the retained figure source. The loss-diagnostic diagram is explicitly schematic. Actual learning curves are retained in the freshly executed notebook.

The guide covers every notebook section. The notebook exercises define which experiments are required and which are optional.

The current PDF uses embedded Times New Roman throughout, black text and monochrome diagrams. Links are black and underlined. The cover links to the current notebook, Colab and course website, with a clickable guide map and contextual official API links.

The actual Times New Roman regular, bold, italic and bold-italic fonts must be available. Windows defaults to C:/Windows/Fonts. On another system, set FLOWMLLAB_FONT_DIR to the directory containing your licensed times.ttf, timesbd.ttf, timesi.ttf and timesbi.ttf files. Fonts are embedded in the PDF and are not redistributed as repository assets. Missing fonts fail the build rather than silently changing the typeface.

Run python validate_pdf.py after rebuilding to check the 21-page guide, Times New Roman text, black text color, monochrome figures, working internal destinations and HTTPS link annotations. Inspect the rendered pages in pdf_qa/ for layout. The compact current audit is format_validation.json.
