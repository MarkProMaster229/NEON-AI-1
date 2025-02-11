import requests
from bs4 import BeautifulSoup
import time

# URL главной страницы форума
url = 'https://2ch.hk/b'

# Отправляем GET-запрос на главную страницу
response = requests.get(url)

# Проверяем успешность запроса
if response.status_code == 200:
    print("Страница успешно загружена")
else:
    print(f"Ошибка загрузки страницы: {response.status_code}")

# Разбираем HTML-страницу
soup = BeautifulSoup(response.text, 'html.parser')

# Ищем все ссылки на треды
thread_links = soup.find_all('a', href=True)

# Фильтруем ссылки, чтобы получить только те, что ведут на треды
thread_urls = [link['href'] for link in thread_links if '/b/res/' in link['href']]

# Ограничиваем количество тредов до 200
thread_urls = thread_urls[:200]

# Печатаем первые 5 ссылок, чтобы убедиться, что правильно все получили
print("Ссылки на треды:", thread_urls[:5])

# Открываем файл для записи
with open("forum_data.txt", "a", encoding="utf-8") as file:
    # Теперь скачиваем сообщения с каждого треда
    for thread_url in thread_urls:
        full_url = f'https://2ch.hk{thread_url}'  # Строим полный URL треда
        thread_response = requests.get(full_url)

        if thread_response.status_code == 200:
            print(f"Тред {thread_url} успешно загружен")
            thread_soup = BeautifulSoup(thread_response.text, 'html.parser')

            # Печатаем первые 500 символов страницы, чтобы проверить структуру
            print(thread_soup.prettify()[:500])

            # Ищем все посты в треде
            posts = thread_soup.find_all('blockquote', class_='postmessage')  # Примерный класс для сообщений
            if not posts:
                print(f"Сообщений не найдено в треде {thread_url}")
            for post in posts:
                post_text = post.get_text()
                file.write(post_text + "\n")  # Сохраняем текст сообщения в файл

            # Добавляем задержку, чтобы не перегружать сервер
            time.sleep(1)
        else:
            print(f"Ошибка загрузки треда {thread_url}")

print("Загрузка и сохранение данных завершены.")
