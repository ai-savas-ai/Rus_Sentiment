import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


class SentimentTester:
    def __init__(self, model_path='./fine_tuned_model'):
        if not os.path.exists(model_path):
            raise Exception(f"Модель не найдена в {model_path}. Убедитесь, что модель обучена и сохранена!")

        print("Загрузка сохраненной модели...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="cpu"
        )
        self.model.eval()  # Переключение в режим оценки
        print("Модель успешно загружена!")

    def analyze_text(self, text):
        """Анализ тональности текста"""
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        # Получение предсказания
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Получение метки класса и уверенности
        label_id = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][label_id].item()

        # Маппинг меток
        sentiment_labels = {
            0: "Нейтральная",
            1: "Позитивная",
            2: "Негативная"
        }

        # Вероятности для всех классов
        probabilities = {
            "Нейтральная": predictions[0][0].item(),
            "Позитивная": predictions[0][1].item(),
            "Негативная": predictions[0][2].item()
        }

        return {
            'sentiment': sentiment_labels[label_id],
            'confidence': confidence,
            'probabilities': probabilities
        }

    def load_texts_from_file(self, file_path):
        """Загрузка текстов из файла"""
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = file.readlines()
        return [text.strip() for text in texts if text.strip()]  # Удаление пустых строк


def test_model():
    try:
        tester = SentimentTester()
        # Загрузка текстов из файла
        test_texts = tester.load_texts_from_file('test.txt')  # Укажите путь к вашему файлу

        print("\nНачало тестирования модели...")
        print("=" * 50)
        for text in test_texts:
            result = tester.analyze_text(text)
            print(f"\nТекст: {text}")
            print(f"Тональность: {result['sentiment']}")
            print(f"Уверенность: {result['confidence']:.2%}")
            print("Вероятности:")
            for tone, prob in result['probabilities'].items():
                print(f"  {tone}: {prob:.2%}")
            print("-" * 50)

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    test_model()