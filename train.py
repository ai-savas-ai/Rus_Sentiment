import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import gc
import matplotlib.pyplot as plt


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label)
        }


def prepare_data(data_path, n_samples=3000):
    print("Загрузка датасета...")
    df = pd.read_csv(data_path)

    class_counts = df['label'].value_counts()
    min_class_count = min(class_counts)

    balanced_dfs = []
    for label in [0, 1, 2]:
        class_df = df[df['label'] == label]
        if len(class_df) > min_class_count:
            class_df = class_df.sample(n=min_class_count, random_state=42)
        balanced_dfs.append(class_df)

    df_balanced = pd.concat(balanced_dfs)

    if len(df_balanced) > n_samples:
        df_balanced = df_balanced.sample(n=n_samples, random_state=42)

    print(f"\nРаспределение классов после балансировки:")
    class_mapping = {0: "Нейтральная", 1: "Позитивная", 2: "Негативная"}
    final_class_counts = df_balanced['label'].value_counts()
    total_samples = len(df_balanced)

    for label, count in final_class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{class_mapping[label]}: {count} образцов ({percentage:.1f}%)")

    texts = df_balanced['text'].values
    labels = df_balanced['label'].values

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.1,
        random_state=42,
        stratify=labels
    )

    print(f"\nИтоговые размеры выборок:")
    print(f"Тренировочная выборка: {len(train_texts)} образцов")
    print(f"Тестовая выборка: {len(test_texts)} образцов")
    return train_texts, test_texts, train_labels, test_labels


def clear_memory():
    """Очистка памяти"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def compute_metrics(pred):
    """Вычисление метрик для оценки модели"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_model(trainer, test_dataset):
    """
    Оценка модели на тестовом наборе данных
    Возвращает и выводит метрики качества
    """
    print("\nОценка модели на тестовом наборе...")

    predictions = trainer.predict(test_dataset)
    labels = predictions.label_ids
    preds = predictions.predictions.argmax(-1)

    metrics = compute_metrics(predictions)

    print("\nМетрики качества модели:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted): {metrics['recall']:.4f}")
    print(f"F1-score (weighted): {metrics['f1']:.4f}")

    print("\nПодробный отчет по классам:")
    class_names = ["Нейтральная", "Позитивная", "Негативная"]
    print(classification_report(labels, preds, target_names=class_names))

    return metrics


def plot_training_metrics(train_losses, eval_metrics):
    epochs = range(len(train_losses))

    # Извлекаем все метрики
    accuracies = [metrics['accuracy'] for metrics in eval_metrics[1:]]
    precisions = [metrics['precision'] for metrics in eval_metrics[1:]]
    recalls = [metrics['recall'] for metrics in eval_metrics[1:]]
    f1_scores = [metrics['f1'] for metrics in eval_metrics[1:]]

    # Создаем график с двумя строками и двумя столбцами
    plt.figure(figsize=(15, 10))

    # График loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # График accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, accuracies, 'r-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # График precision и recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, precisions, 'g-', label='Precision')
    plt.plot(epochs, recalls, 'y-', label='Recall')
    plt.title('Precision and Recall (weighted)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # График F1-score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, f1_scores, 'm-', label='F1-score')
    plt.title('F1-Score (weighted)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_model():
    print("Загрузка модели и токенизатора...")
    model_name = "blanchefort/rubert-base-cased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model without device_map parameter
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Модель загружена на GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU недоступен, используется CPU")

    train_texts, test_texts, train_labels, test_labels = prepare_data('_sentiment_.csv')

    print("Создание датасетов...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir='../RusSentiment/results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Remove no_cuda=True to enable GPU usage
        dataloader_num_workers=0,
        logging_first_step=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Начало обучения...")
    train_losses = []
    eval_metrics = []

    # Начальная оценка
    initial_metrics = evaluate_model(trainer, test_dataset)
    eval_metrics.append(initial_metrics)

    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\nЭпоха {epoch + 1}/{int(training_args.num_train_epochs)}")

        # Обучение
        train_result = trainer.train()

        # Получаем loss из истории обучения
        train_loss = train_result.metrics.get('train_loss', 0.0)
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss:.4f}")

        # Оценка
        metrics = evaluate_model(trainer, test_dataset)
        eval_metrics.append(metrics)

        # Вывод текущих метрик
        print(f"Evaluation metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")

    # Построение графиков
    plot_training_metrics(train_losses, eval_metrics)

    print("Сохранение модели...")
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

    return model, tokenizer, eval_metrics[-1]


def evaluate_text(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    label_id = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][label_id].item()

    sentiment_labels = {
        0: "Нейтральная",
        1: "Позитивная",
        2: "Негативная"
    }

    probabilities = {
        "Нейтральная": predictions[0][0].item(),
        "Позитивная": predictions[0][1].item(),
        "Негативная": predictions[0][2].item()
    }

    return sentiment_labels[label_id], confidence, probabilities


def main():
    try:
        # Обучение модели
        print("Запуск обучения модели...")
        model, tokenizer, metrics = train_model()

        # Интерактивный режим
        print("\nМодель готова к работе!")
        print("Введите текст для анализа (или 'exit' для выхода)")

        while True:
            text = input("\nВведите текст: ")

            if text.lower() == 'exit':
                print("Работа завершена")
                break

            # Очистка памяти перед предсказанием
            clear_memory()

            # Получение предсказания
            sentiment, confidence, probabilities = evaluate_text(model, tokenizer, text)
            print(f"\nОсновная тональность: {sentiment}")
            print(f"Уверенность: {confidence:.2%}")
            print("\nВероятности для всех классов:")
            for tone, prob in probabilities.items():
                print(f"{tone}: {prob:.2%}")

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    finally:
        clear_memory()


if __name__ == "__main__":
    main()