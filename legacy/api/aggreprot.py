from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup  # Для парсинга HTML-ответа
import time  # Для возможных коротких пауз


# --- 1. Настройка Selenium WebDriver ---
def setup_driver():
    """
    Настраивает и возвращает экземпляр WebDriver для Chrome.
    webdriver_manager автоматически скачивает и устанавливает ChromeDriver.
    """
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()

    # Дополнительные опции для браузера:
    # options.add_argument("--headless") # Запускать браузер в фоновом режиме (без графического интерфейса)
    options.add_argument("--no-sandbox")  # Обязательно для некоторых сред (например, Docker)
    options.add_argument("--disable-dev-shm-usage")  # Для избежания проблем в Linux-контейнерах

    # Можно добавить User-Agent, чтобы сайт воспринимал запрос как от обычного браузера
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)  # Неявное ожидание до 10 секунд при поиске элементов
    return driver


# --- 2. Основная функция для заполнения формы и получения результатов ---
def fill_and_submit_form(driver_instance, form_page_url, sequence_data):
    """
    Заполняет форму на сайте, нажимает кнопку Submit и возвращает HTML новой страницы.

    :param driver_instance: Экземпляр Selenium WebDriver.
    :param form_page_url: Полный URL страницы, на которой расположена форма.
                           Например: "https://your-website.com/form_page.html"
    :param sequence_data: Строка с данными для поля 'sequence'.
    :return: HTML-содержимое страницы после отправки формы.
    """
    print(f"Открытие страницы формы: {form_page_url}")
    driver_instance.get(form_page_url)

    try:
        # 1. Поиск текстового поля (textarea) по его атрибуту 'name'
        # Используем WebDriverWait для явного ожидания, пока элемент станет доступным.
        textarea_field = WebDriverWait(driver_instance, 15).until(
            EC.presence_of_element_located((By.NAME, "sequence"))
        )
        print("Текстовое поле 'sequence' найдено.")

        # 2. Ввод данных в текстовое поле
        textarea_field.send_keys(sequence_data)
        print("Данные введены в текстовое поле.")

        # Дополнительная небольшая пауза, если сайт имеет медленную реакцию
        # time.sleep(1)

        # 3. Поиск кнопки "Submit"
        # Ищем кнопку по ее типу и значению атрибута 'value'.
        submit_button = WebDriverWait(driver_instance, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@type='submit' and @value='submit!']"))
            # Альтернативный вариант, если value может меняться или его нет:
            # EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit']"))
        )
        print("Кнопка 'submit!' найдена.")

        # 4. Нажатие кнопки "Submit"
        submit_button.click()
        print("Кнопка 'submit!' нажата. Ожидание загрузки новой страницы...")

        # 5. Ожидание загрузки новой страницы с результатами
        # Здесь вам нужно указать, какой элемент должен появиться на СТРАНИЦЕ РЕЗУЛЬТАТОВ,
        # чтобы Selenium понял, что страница загружена.
        # Замените 'some_unique_id_on_result_page' на реальный ID или селектор.
        # Например, это может быть <h1> с заголовком результатов, или <div> с контейнером данных.
        WebDriverWait(driver_instance, 30).until(
            EC.url_contains("/cgi-bin/aap/aap_ov.pl") # Ожидание, что URL изменится на action-URL формы
            # EC.title_contains("Results") # Ожидание, что заголовок страницы содержит "Results"
        )
        print("Страница результатов загружена.")

        # 6. Получение всего HTML-содержимого новой страницы
        new_page_html = driver_instance.page_source
        return new_page_html

    except Exception as e:
        print(f"Произошла ошибка во время взаимодействия с сайтом: {e}")
        return None


# --- 3. Вызов функций и обработка результатов ---
if __name__ == "__main__":
    # 🚨🚨🚨 ОЧЕНЬ ВАЖНО: Замените этот URL на реальный адрес СТРАНИЦЫ, ГДЕ НАХОДИТСЯ ВАША ФОРМА 🚨🚨🚨
    # Например: "https://your-website.com/analysis_tool.html"
    # ЭТО НЕ ДОЛЖЕН БЫТЬ ТОЛЬКО action-URL из формы (/cgi-bin/aap/aap_ov.pl)
    # Это URL страницы, которую пользователь открывает в браузере, чтобы увидеть форму.
    target_form_url = "http://bioinf.uab.es/aggrescan/"

    # 🚨🚨🚨 ОЧЕНЬ ВАЖНО: Замените "some_unique_id_on_result_page" на актуальный ID
    # элемента на странице с результатами, который появляется после отправки формы. 🚨🚨🚨
    # Например, если после отправки формы на странице появляется <div id="analysis_output">,
    # то используйте "analysis_output".
    # Без этого Selenium не будет знать, что новая страница загрузилась.
    # Если на странице результатов нет уникального ID, возможно, вам придется изменить логику ожидания
    # (например, ждать, пока URL изменится на action-URL, или ждать появления определенного текста).

    # Пример данных для отправки
    my_sequence_data = """
>RPL27_human
MGKFMKPGKVVLVLAGRYSGRKAVIVKNIDDGTSDRPYSHALVAGIDRYPRKVTAAMGKK
KIAKRSKIKSFVKVYNYNHLMPTRYSVDIPLDKTVVNKDVFRDPALKRKARREAKVKFEE
RYKTGKNKWFFQKLRF
>RPL27_human_Y75P | Direct mutation
MGKFMKPGKVVLVLAGRYSGRKAVIVKNIDDGTSDRPYSHALVAGIDRYPRKVTAAMGKK
KIAKRSKIKSFVKVPNYNHLMPTRYSVDIPLDKTVVNKDVFRDPALKRKARREAKVKFEE
RYKTGKNKWFFQKLRF
>RPL27_human_Y75G | Direct mutation
MGKFMKPGKVVLVLAGRYSGRKAVIVKNIDDGTSDRPYSHALVAGIDRYPRKVTAAMGKK
KIAKRSKIKSFVKVGNYNHLMPTRYSVDIPLDKTVVNKDVFRDPALKRKARREAKVKFEE
RYKTGKNKWFFQKLRF
>RPL27_human_Y75D | Direct mutation
MGKFMKPGKVVLVLAGRYSGRKAVIVKNIDDGTSDRPYSHALVAGIDRYPRKVTAAMGKK
KIAKRSKIKSFVKVDNYNHLMPTRYSVDIPLDKTVVNKDVFRDPALKRKARREAKVKFEE
RYKTGKNKWFFQKLRF
>RPL27_human_Y75K | Direct mutation
MGKFMKPGKVVLVLAGRYSGRKAVIVKNIDDGTSDRPYSHALVAGIDRYPRKVTAAMGKK
KIAKRSKIKSFVKVKNYNHLMPTRYSVDIPLDKTVVNKDVFRDPALKRKARREAKVKFEE
RYKTGKNKWFFQKLRF
"""

    driver = None  # Инициализируем driver вне блока try, чтобы он был доступен в finally
    try:
        driver = setup_driver()

        # Запускаем процесс заполнения и отправки формы
        result_page_html = fill_and_submit_form(driver, target_form_url, my_sequence_data)

        if result_page_html:
            print("\n--- HTML-содержимое страницы результатов (первые 1000 символов) ---")
            print(result_page_html)
            print("...")
            print("\n--- Конец HTML-ответа ---")

            # Здесь вы можете добавить код для парсинга 'result_page_html' с помощью BeautifulSoup
            # и извлечения нужных данных.
            # Пример:
            # soup = BeautifulSoup(result_page_html, 'lxml')
            # result_element = soup.find('div', id='some_result_id') # Ищем элемент по его ID на странице результатов
            # if result_element:
            #     print(f"\nИзвлеченные данные: {result_element.get_text().strip()}")
            # else:
            #     print("\nНе удалось найти элемент результатов на странице.")

        else:
            print("Не удалось получить HTML-содержимое страницы результатов.")

    except Exception as e:
        print(f"Общая ошибка выполнения скрипта: {e}")
    finally:
        if driver:
            driver.quit()  # Очень важно всегда закрывать браузер
            print("Браузер закрыт.")

