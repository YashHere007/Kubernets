from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from transformers import MarianMTModel, MarianTokenizer
from doctr.models import ocr_predictor
import sys
from PIL import Image
import time

class Worker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, image_path, translation_model, translation_tokenizer, ocr_model):
        super().__init__()
        self.image_path = image_path
        self.translation_model = translation_model
        self.translation_tokenizer = translation_tokenizer
        self.ocr_model = ocr_model

    def run(self):
        start_time = time.time()

        # Load and process the image
        image = Image.open(self.image_path)
        image = image.resize((1024, 1024))  # Resize to improve processing speed
        # Directly use the image in DocTR OCR predictor
        result = self.ocr_model([image])

        # Extract text from DocTR result
        extracted_text = ' '.join([word['value'] for block in result.pages[0].blocks for line in block.lines for word in line.words])

        # Translate text if it's not English
        translated_text = self.translate_text(extracted_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        self.result_ready.emit(translated_text)

    def translate_text(self, text):
        # Tokenize and translate
        inputs = self.translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.translation_model.generate(**inputs)
        translated_text = [self.translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return ' '.join(translated_text)

class TextExtractorTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_models()

    def init_ui(self):
        self.setWindowTitle('Text Extractor and Translator')
        self.setGeometry(100, 100, 800, 600)

        # Layout and Widgets
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText('Extracted and Translated Text Will Appear Here')
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        
        layout.addWidget(self.load_button)
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_models(self):
        # Load MarianMT model and tokenizer
        self.translation_model_name = 'Helsinki-NLP/opus-mt-mul-en'
        self.translation_model = MarianMTModel.from_pretrained(self.translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(self.translation_model_name)
        
        # Load DocTR model
        self.ocr_model = ocr_predictor(pretrained=True)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Images (*.png *.jpg *.jpeg)')
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.List)

        if file_dialog.exec_():
            image_files = file_dialog.selectedFiles()
            for image_file in image_files:
                self.process_image(image_file)

    def process_image(self, image_path):
        self.worker = Worker(image_path, self.translation_model, self.translation_tokenizer, self.ocr_model)
        self.worker.result_ready.connect(self.display_result)
        self.worker.start()

    def display_result(self, text):
        self.text_edit.setPlainText(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextExtractorTranslatorApp()
    window.show()
    sys.exit(app.exec_())
