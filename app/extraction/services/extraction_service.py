import base64
import cv2
import easyocr
import numpy as np
from app.services.base_service import BaseService


class ExtractionService(BaseService):
    async def preprocess_image(self, image_bytes: bytearray) -> tuple[cv2.typing.MatLike, 
                                                                         cv2.typing.MatLike]:
        """
        Preprocess image for better text recognition
        - Convert to grayscale
        - Apply noise reduction
        - Apply adaptive thresholding
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        return image, binary

    async def draw_text_regions(self, original_image: cv2.typing.MatLike, 
                                text_regions: list[tuple[int,int]]) -> cv2.typing.MatLike:
        """
        Draw rectangles around detected text regions
        """
        result_image = original_image.copy()
        
        for text_region in text_regions:
            (top_left, bottom_right) = text_region
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            
            # Рисуем прямоугольник
            cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)

        return result_image
    
    async def extract_text_and_regions(self, binary: cv2.typing.MatLike) -> tuple[str, list[tuple[int,int]]]:
        """
        Extract text and text regions from image
        """
        # Инициализируем EasyOCR
        reader = easyocr.Reader(lang_list = ['ru', 'en'], gpu = False)
        
        # Распознаем текст
        results = reader.readtext(binary)
        
        text = ''
        text_regions = []

        # Собираем весь распознанный текст
        for result in results:
            text += result[1] + "\n"
            text_regions.append(tuple([result[0][0], result[0][2]]))
        
        return text, text_regions
    
    async def convert_image_to_base64_string(self, image: cv2.typing.MatLike) -> str:
        """
        Convert image to base64 string
        """

        _, buffer = cv2.imencode('.png', image)

        return base64.b64encode(buffer).decode('utf-8')

    

extraction_service = ExtractionService()