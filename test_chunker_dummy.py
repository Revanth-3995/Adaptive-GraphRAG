import fitz
import os

# create a dummy pdf
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "This is a test document. " * 100)
doc.save("test_dummy.pdf")
doc.close()
