file = "https://bioinfo.crbm.cnrs.fr/tools/ArchCandy-2/datastore/OoTHWFR.csv"

fl = "/datastore/6930a71db94f6.json"
f = "https://bioinfo.crbm.cnrs.fr/tools/Cross-Beta_pred/datastore/"


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
def setup_driver():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)  # Неявное ожидание до 10 секунд при поиске элементов
    return driver


def fill_and_submit_form(driver_instance, form_page_url, sequence_data):
    print(f"Открытие страницы формы: {form_page_url}")
    driver_instance.get(form_page_url)

    try:
        textarea_field = WebDriverWait(driver_instance, 15).until(
            EC.presence_of_element_located((By.NAME, "sequence"))
        )
        print("Текстовое поле 'sequence' найдено.")

        textarea_field.send_keys(sequence_data)
        print("Данные введены в текстовое поле.")

        submit_button = WebDriverWait(driver_instance, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
        )
        print("Кнопка 'submit!' найдена.")

        submit_button.click()
        print("Кнопка 'submit!' нажата. Ожидание загрузки новой страницы...")
        WebDriverWait(driver_instance, 30)
        print("Страница результатов загружена.")
        new_page_html = driver_instance.page_source
        return new_page_html

    except Exception as e:
        print(f"Произошла ошибка во время взаимодействия с сайтом: {e}")
        return None


if __name__ == "__main__":
    target_form_url = "https://bioinfo.crbm.cnrs.fr/index.php?route=tools&tool=35"

    my_sequence_data = """
    GAVPVQIVYKGADRTVAKIDKPLQAVVILFH
    """

    driver = None
    try:
        driver = setup_driver()

        result_page_html = fill_and_submit_form(driver, target_form_url, my_sequence_data)

        if result_page_html:
            datajson =result_page_html.split("jsonData")[1][2:][3:].split("\"")[0]
            print(datajson)

            data_url = f"{f}{datajson}.json"
        else:
            print("Не удалось получить HTML-содержимое страницы результатов.")

    except Exception as e:
        print(f"Общая ошибка выполнения скрипта: {e}")
    finally:
        if driver:
            driver.quit()  # Очень важно всегда закрывать браузер
            print("Браузер закрыт.")

