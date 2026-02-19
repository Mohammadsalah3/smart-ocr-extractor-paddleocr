import gradio as gr
from paddleocr import PaddleOCR
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np

POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"

#PaddleOCR models
OCR_MODELS = {
    "eng": PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    ),
    "ar": PaddleOCR(
        lang="ar",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    ),
}

# Preprocessing 
def preprocess_image(pil_image, steps):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if "Grayscale" in steps:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if "Resize" in steps:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if "Denoise" in steps:
        img = cv2.medianBlur(img, 3)

    if "Deskew" in steps:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            angle = -angle

            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Contrast
    if "Contrast" in steps:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Ensure uint8 contiguous
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img = np.ascontiguousarray(img)

    # Return PIL RGB for Gradio display / next steps
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

#  Handle Upload 
def handle_upload(file):
    if file is None:
        return None, None

    filename = file.name.lower()

    if filename.endswith(".pdf"):
        pages = convert_from_path(file.name, dpi=300, poppler_path=POPPLER_PATH)
        return pages[0].convert("RGB"), pages
    else:
        img = Image.open(file.name).convert("RGB")
        return img, None

# Apply Changes (Preview) 
def apply_changes(original_image, pdf_pages, steps):
    if original_image is None:
        return None

    if pdf_pages:
        return preprocess_image(pdf_pages[0].convert("RGB"), steps)
    return preprocess_image(original_image, steps)

# Paddle OCR runner 
def run_paddle_on_pil(pil_img, lang_choice):
    img_np = np.array(pil_img)

    # PaddleX usually expects BGR uint8 contiguous
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)

    def extract_from_model(model):
        try:
            result = model.predict(img_np)
        except Exception as e:
            return [f"[PaddleOCR error] Details: {type(e).__name__}: {e}"]

        lines = []
        for res in result:
            if "rec_texts" in res:
                lines.extend(res["rec_texts"])
        return lines

    if lang_choice in ("eng", "ar"):
        lines = extract_from_model(OCR_MODELS[lang_choice])
        return "\n".join([t for t in lines if str(t).strip()]).strip()

    # mix: run both and merge unique
    lines_en = extract_from_model(OCR_MODELS["eng"])
    lines_ar = extract_from_model(OCR_MODELS["ar"])

    merged, seen = [], set()
    for t in lines_en + lines_ar:
        t = str(t).strip()
        if t and t not in seen:
            merged.append(t)
            seen.add(t)

    return "\n".join(merged).strip()

# Extract Text 
def extract_text(original_image, pdf_pages, steps, lang_choice):
    if original_image is None:
        return "Upload a file first."

    if pdf_pages:
        full_text = []
        for page in pdf_pages:
            processed = preprocess_image(page.convert("RGB"), steps)
            page_text = run_paddle_on_pil(processed, lang_choice)
            if page_text:
                full_text.append(page_text)
        return "\n\n".join(full_text).strip()

    processed = preprocess_image(original_image, steps)
    return run_paddle_on_pil(processed, lang_choice)

# UI 
def main():
    with gr.Blocks(title="Smart OCR Extractor (PaddleOCR)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Smart OCR Extractor (PaddleOCR)")
    
        pdf_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                upload_box = gr.File(
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    label="Upload PDF or Image"
                )

                lang_choice = gr.Dropdown(
                    choices=["eng", "ar", "mix"],
                    value="eng",
                    label="Language"
                )

                preprocessing_steps = gr.CheckboxGroup(
                    choices=["Grayscale", "Resize", "Denoise", "Deskew", "Contrast"],
                    value=["Grayscale", "Resize"],
                    label="Preprocessing"
                )

                apply_btn = gr.Button("Apply Changes (Preview)")
                extract_btn = gr.Button("Extract Text")

            with gr.Column(scale=1):
                original_preview = gr.Image(label="Original Preview")
                processed_preview = gr.Image(label="Processed Preview")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Extracted Text", lines=25)

        upload_box.change(
            handle_upload,
            inputs=upload_box,
            outputs=[original_preview, pdf_state]
        )

        apply_btn.click(
            apply_changes,
            inputs=[original_preview, pdf_state, preprocessing_steps],
            outputs=processed_preview
        )

        extract_btn.click(
            extract_text,
            inputs=[original_preview, pdf_state, preprocessing_steps, lang_choice],
            outputs=output_text
        )

    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()
