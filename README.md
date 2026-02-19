![Python](https://img.shields.io/badge/Python-3.11-blue)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-DeepLearning-green)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-red)
![Gradio](https://img.shields.io/badge/Gradio-WebApp-orange)

# Smart OCR Extractor (PaddleOCR)

> Intelligent Document OCR with Preprocessing, Deskewing, Contrast Enhancement & Multilingual Support

An advanced OCR web application built with **PaddleOCR, OpenCV, and Gradio** that enables accurate text extraction from PDFs and images through an optimized preprocessing pipeline.


## Features
- ✅ Upload **PDF / PNG / JPG**
- ✅ Convert PDF pages to images using **pdf2image + Poppler**
- ✅ Preprocessing pipeline:
  - Grayscale
  - Resize (2x)
  - Denoise (median blur)
  - Deskew (auto-rotation)
  - Contrast enhancement (CLAHE)
- ✅ Preview: original + processed
- ✅ OCR language modes:
  - `eng`
  - `ar`
  - `mix` (runs both + merges output)
- ✅ Multi-page PDF extraction

