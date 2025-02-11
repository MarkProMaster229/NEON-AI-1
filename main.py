import os
import torch
import torch.nn as nn
import torch.optim as optim


file_path = r"C:\Users\chelovek\PycharmProjects\NEON-AI-1\data.txt"#пара вопрос/ответ
model_path = "chatbot_model.pth"  # Путь для сохранения/загрузки модели

print("Модель загружена!")

# Проверяем существование файла
if not os.path.exists(file_path):
    print("Ошибка: Файл data.txt не найден!")
    exit()

# Чтение и обработка данных
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

lines = [line.strip() for line in lines if line.strip()]

def clean_text(text):
    # Убираем знаки препинания и приводим к нижнему регистру
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower()

# Разделение на вопросы и ответы(я)
questions = []
answers = []

for i in range(len(lines) - 1):
    if lines[i].startswith("Пользователь:"):
        questions.append(lines[i].replace("Пользователь: ", ""))
        if i + 1 < len(lines) and lines[i + 1].startswith("Бот:"):
            answers.append(lines[i + 1].replace("Бот: ", ""))

# Вывод пар вопрос-ответ
for q, a in zip(questions, answers):
    print(f"Вопрос: {q} → Ответ: {a}")

print(f"\nНайдено {len(questions)} пар 'вопрос-ответ'.")

# Создание словаря
word_to_index = {}
index_to_word = {}
all_words = set()

for sentence in questions + answers:
    all_words.update(sentence.split())

for i, word in enumerate(sorted(all_words)):
    word_to_index[word] = i + 1  # Индексация с 1
    index_to_word[i + 1] = word

# Функция токенизации
def text_to_sequence(text):
    return [word_to_index[word] for word in text.split() if word in word_to_index]

# Токенизация данных
tokenized_questions = [text_to_sequence(q) for q in questions]
tokenized_answers = [text_to_sequence(a) for a in answers]
# Проверка слов в словаре
for sentence in questions + answers:
    for word in sentence.split():
        if word not in word_to_index:
            print(f"Слово не найдено в словаре: {word}")

# Вывод примеров токенизации
for q, t_q, a, t_a in zip(questions, tokenized_questions, answers, tokenized_answers):
    print(f"Вопрос: {q} → {t_q}")
    print(f"Ответ: {a} → {t_a}\n")

print(f"Всего уникальных слов: {len(word_to_index)}")

# Блок паддинга
max_seq_length = max(max(len(seq) for seq in tokenized_questions + tokenized_answers), 1)

def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) for seq in sequences]

padded_questions = pad_sequences(tokenized_questions, max_seq_length)
padded_answers = pad_sequences(tokenized_answers, max_seq_length)

# Архитектура модели
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

# Параметры модели
VOCAB_SIZE = len(word_to_index) + 1
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128

# Создание модели
model = ChatbotModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE
)

# Загрузка модели, если она уже обучена
if os.path.exists(model_path):
    print("Загрузка сохраненной модели...")
    model.load_state_dict(torch.load(model_path))
    print("Модель загружена!")
else:
    print("Обучение новой модели...")
    # Преобразование в тензоры
    inputs = torch.tensor(padded_questions, dtype=torch.long)
    targets = torch.tensor(padded_answers, dtype=torch.long)

    print(f"Размерность данных: Входы - {inputs.shape}, Цели - {targets.shape}")

    # Обучение модели
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 500
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(inputs)

        # Изменяем размерности для функции потерь
        outputs = outputs.view(-1, VOCAB_SIZE)  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]

        # Вычисление потерь
        loss = criterion(outputs, targets)

        # Обратное распространение и обновление весов
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.4f}")

    # Сохранение модели после обучения
    torch.save(model.state_dict(), model_path)
    print("Модель сохранена!")

print("Обучение завершено!")

# Переводим модель в режим оценки
model.eval()

# Функция для генерации ответа
def generate_response(question):
    print(f"Исходный вопрос: {question}")  # Проверяем вопрос перед токенизацией
    tokenized = text_to_sequence(question)
    print(f"Токенизированный вопрос: {tokenized}")  # Проверяем токенизацию
    padded = pad_sequences([tokenized], max_seq_length)
    input_tensor = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_indices = torch.argmax(output, dim=-1).squeeze().tolist()

    print(f"Предсказанные индексы: {predicted_indices}")  # Проверяем индексы

    response_words = []
    for idx in predicted_indices:
        if idx == 0:
            continue
        if idx in index_to_word:
            response_words.append(index_to_word[idx])
        else:
            response_words.append("<UNK>")

    response = " ".join(response_words)
    print(f"Ответ: {response}")  # Проверяем результат
    return response



#говорить -
new_question = "Что ты умеешь?"
response = generate_response(new_question)
print(f"Вопрос: {new_question} → Ответ: {response}")