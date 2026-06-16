import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    scipy_stats = None
    SCIPY_AVAILABLE = False


BOOKING_COLUMNS = [
    "hotel_id",
    "booking_created_date",
    "is_STR",
    "gbb",
    "roomnights",
    "sales_volumes_rub",
    "revenue_rub"
]

ACTION_COLUMNS = [
    "action_date",
    "subject",
    "outcome",
    "hotel_id"
]

METRICS = {
    "gbb": "GBB / количество бронирований",
    "roomnights": "Roomnights / количество ночей",
    "sales_volumes_rub": "Sales volume / объём продаж",
    "revenue_rub": "Revenue / доход объекта"
}

METRIC_DESCRIPTIONS = {
    "gbb": "Главная метрика по ТЗ. Показывает количество бронирований объекта.",
    "roomnights": "Дополнительная метрика. Показывает количество забронированных ночей.",
    "sales_volumes_rub": "Дополнительная метрика. Показывает общий объём продаж в рублях.",
    "revenue_rub": "Дополнительная метрика. Показывает доход объекта от бронирований. Это не комиссия платформы."
}

MAIN_OUTCOMES = ["Published", "Fixed", "Returned", "Refused"]

SUCCESS_OUTCOMES = [
    "Published", "Fixed", "Returned", "Resolved",
    "Positively resolved", "Checked-all good"
]

NEGATIVE_OUTCOMES = [
    "Refused", "Impossible to fix", "Negatively resolved"
]

IN_PROGRESS_OUTCOMES = [
    "Needs follow-up", "Pending"
]

SUBJECT_DESCRIPTIONS = {
    "ECLC": "Early Check-in / Late Check-out: добавление опции раннего заезда и позднего выезда.",
    "Adding meals": "Добавление питания в тариф.",
    "New room categories": "Добавление новых категорий номеров.",
    "Availability improvement": "Улучшение доступности, обычно не менее 30 дней.",
    "B2B rates": "Заведение спецтарифов для закрытых B2B-каналов.",
    "Bedding correction": "Исправление конфигурации кроватей в Extranet.",
    "Cancellation policy improvement": "Улучшение политики отмены в тарифах.",
    "Higher commission": "Обсуждение договора на повышенную комиссию.",
    "Hybrid": "Сотрудничество по гибридной модели.",
    "Net rates": "Добавление нетто-тарифов.",
    "Net rates improvement": "Улучшение нетто-тарифов в цене.",
    "Non-refundable rate": "Заведение невозвратных тарифов.",
    "Opaque rate": "Заведение тарифов для постоянных пользователей.",
    "Rate without meals": "Заведение дополнительных тарифов без питания.",
    "Rates improvement": "Улучшение текущих тарифов в цене.",
    "B2B rates improvement": "Улучшение B2B-тарифов в цене.",
    "Special offer": "Заведение спецпредложения.",
    "Content issues": "Проблемы, связанные с контентом.",
    "Package rates": "Добавление пакетных тарифов.",
    "Parity": "Корректировка лучшей цены в сравнении с конкурентами.",
    "Top Stays": "Подключение программы лояльности Top Stays.",
    "VCC": "Подключение оплат через VCC-карты.",
    "GURU 1": "Добавление тарифов программы лояльности GURU 1 уровень.",
    "GURU 2": "Добавление тарифов программы лояльности GURU 2 уровень.",
    "GURU 3": "Добавление тарифов программы лояльности GURU 3 уровень.",
    "GURU 4": "Добавление тарифов программы лояльности GURU 4 уровень.",
    "Parity Yandex": "Корректировка лучшей цены в сравнении с Я.Путешествия.",
    "Parity Hotel's Website": "Корректировка лучшей цены в сравнении с сайтом отеля.",
    "Guru bonus": "Добавление тарифов программы лояльности GURU бонусы.",
    "GEO-rates": "Добавление гео-тарифов.",
    "Commission Override": "Подключение к программе лояльности Commission Override.",
    "Retention": "Предотвращение отключения объекта от платформы.",
    "Top Position": "Продажа топовой позиции объекту.",
    "Commercial meeting": "Коммуникация с отельером для поддержания деловых отношений."
}

MONTH_NAMES = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}

MONTH_ORDER = list(MONTH_NAMES.values())

GLOSSARY = {
    "GBB": "Gross Booked Bookings – количество бронирований. Главная метрика проекта.",
    "Roomnights": "Количество забронированных ночей. Дополнительная метрика глубины спроса.",
    "Sales volume": "Объём продаж в рублях. Показывает денежный масштаб бронирований.",
    "Revenue": "Доход объекта от бронирований. В рамках текущих данных это не комиссия платформы.",
    "Action Log": "Запись о коммуникации или действии менеджера по объекту.",
    "Subject": "Тип действия менеджера: тарифы, доступность, контент, программа лояльности и т.д.",
    "Outcome": "Результат действия: Published, Fixed, Returned, Refused и другие статусы.",
    "STR": "Short-Term Rental / апарт-объект. В дашборде это отдельный тип объекта.",
    "HOTEL": "Обычный отель. В дашборде сравнивается отдельно от STR.",
    "Seasonality index": "Сезонный индекс месяца. Показывает, насколько месяц сильнее или слабее среднего месяца.",
    "Expected value": "Ожидаемое значение метрики для объекта с учётом его базового уровня и сезонности.",
    "Efficiency": "Факт / ожидание. Значение 1.00 означает ровно по сезонному ожиданию.",
    "Individual effect": "Отклонение конкретного объекта от фона похожих объектов.",
    "Manager effect": "Изменение объекта после Action Log минус изменение похожих объектов за тот же период.",
    "Peer group": "Похожие объекты внутри одной локалии: STR сравнивается с STR, HOTEL – с HOTEL.",
}

METHODOLOGY_LINKS = {
    "seasonality": "#method-seasonality",
    "expected": "#method-expected",
    "efficiency": "#method-efficiency",
    "individual": "#method-individual-effect",
    "action_window": "#method-action-window",
    "overlap": "#method-overlap",
    "manager": "#method-manager-effect",
    "correlation": "#method-correlation",
    "metrics": "#method-four-metrics",
    "ttest": "#method-ttest",
    "clustering": "#method-clustering",
}
