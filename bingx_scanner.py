import os
import time
import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import requests

# Импорт клиента BingxClient из предоставленного модуля
from bingx_client import BingxClient

# ==============================================================================
# --- НАСТРОЙКИ СКАНИРОВАНИЯ ---
# ==============================================================================
CHAT_ID = 1234 #https://t.me/Get_myidrobot
BOT_TOKEN = '' # @botFather

#BINGX API - https://bingx.com/en/accounts/api 
#регистрация с бонусами и моей личной поддержкой - https://bingxdao.com/invite/68FANLWHZ/
BINGX_API_KEY = '' 
BINGX_SECRET = '' 


SCAN_INTERVAL_SECONDS: int = 60      # Интервал сканирования в секундах
TIMEFRAME: str = "1h"                # Таймфрейм для свечей индикаторов (15m, 1h, 4h)
KLINES_LIMIT: int = 1000              # Количество свечей для расчёта EMA и MACD

# --- ПОРОГИ ОСНОВНЫХ ФИЛЬТРОВ ---
OI_THRESHOLD: float = 1_000_000.0    # Минимальный Open Interest в USDT
OI_CHANGE_PERCENT: float = 5.0      # Минимальный рост OI в процентах (за последние 24 часа с Binance)
FUNDING_RATE_THRESHOLD: float = 0.001 # 0.1% (абсолютное значение, напр. 0.005 = 0.5%)

# --- НАСТРОЙКИ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ ---
EMA_FAST: int = 9
EMA_SLOW: int = 21

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# --- РЕЖИМЫ РАБОТЫ ---
USE_OI_CHANGE: bool = True          # True = процентное изменение OI, False = абсолютный порог OI
OI_CHANGE_WINDOW: int = 24           # Окно свечей для изменения OI (если USE_OI_CHANGE = True)
FILTER_MODE: str = "AND"             # "AND" = оба фильтра выполняются, "OR" = хотя бы один

# --- НАСТРОЙКИ ТЕЛЕГРАМ ---
TELEGRAM_COOLDOWN_HOURS: float = 4.0 # Кулдаун повторной отправки сигнала (в часах)

# --- ПАРАМЕТРЫ ТОРГОВЛИ И МОНИТОРИНГА ---
TRADE_ENABLED: bool = True           # True = открыть позицию на бирже при сигнале, False = только сигналы
TRADE_SIZE_USDT: float = 100.0        # Объем сделки в USDT (маржа)
LEVERAGE: int = 15                # Кредитное плечо
TP_PERCENT: float = 20.0              # Тейк-Профит в процентах от точки входа (2.0 = 2%)
SL_PERCENT: float = 5.0            # Стоп-Лосс в процентах от точки входа (1.5 = 1.5%)
TRAILING_ACTIVATION_PERCENT: float = 2.0  # Процент прибыли для активации трейлинга (1.0 = 1%)
TRAILING_RATE_PERCENT: float = 3.0   # Callback rate трейлинга в процентах (0.5 = 0.5%)

# Новые переменные для мониторинга фандинга
funding_min: float = 0.0003          # Минимальная ставка финансирования (например, -0.3%)
funding_max: float = -0.0003           # Максимальная ставка финансирования (например, +0.3%)

MONITORING_INTERVAL_SECONDS: int = 300  # Интервал мониторинга позиций (5 минут = 300 секунд)
ONE_WAY_MODE: bool = False           # Режим One-way (Hedge Mode по умолчанию)
# ==============================================================================


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Преобразует строку таймфрейма (напр. '15m', '1h', '1d') в количество минут.
    
    Args:
        timeframe (str): Таймфрейм в виде строки.
        
    Returns:
        int: Количество минут.
    """
    try:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
    except Exception:
        pass
    return 60  # Значение по умолчанию


# --- ИНТЕГРАЦИЯ С BINANCE ДЛЯ АНАЛИЗА HISTORICAL OPEN INTEREST ---
BINANCE_FAPI_URL = "https://fapi.binance.com"
REQUEST_TIMEOUT: int = 10

def binance_get(endpoint: str, params: Optional[dict] = None) -> Any:
    url = BINANCE_FAPI_URL + endpoint
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_oi_hist(symbol: str, limit: int) -> List[Dict[str, Any]]:
    # Binance-совместимое имя символа (без дефисов, например, BTCUSDT вместо BTC-USDT)
    binance_symbol = symbol.replace("-", "")
    return binance_get("/futures/data/openInterestHist", {"symbol": binance_symbol, "period": "5m", "limit": limit})

def get_binance_oi_change_24h(symbol: str) -> Optional[float]:
    """
    Рассчитывает процент изменения OI за последние 24 часа (288 свечей по 5м) с Binance.
    """
    try:
        hist = get_oi_hist(symbol, 288)
        if not hist or len(hist) < 2:
            return None
        oldest_val = float(hist[0].get("sumOpenInterestValue") or hist[0].get("sumOpenInterest") or 0.0)
        latest_val = float(hist[-1].get("sumOpenInterestValue") or hist[-1].get("sumOpenInterest") or 0.0)
        if oldest_val == 0.0:
            return None
        return ((latest_val - oldest_val) / oldest_val) * 100.0
    except Exception as e:
        print(f"[BINANCE OI 24H ERROR] {symbol}: {e}")
        return None

def get_binance_oi_change_1h(symbol: str) -> Optional[float]:
    """
    Рассчитывает процент изменения OI за последний час (12 свечей по 5м) с Binance.
    """
    try:
        hist = get_oi_hist(symbol, 12)
        if not hist or len(hist) < 2:
            return None
        oldest_val = float(hist[0].get("sumOpenInterestValue") or hist[0].get("sumOpenInterest") or 0.0)
        latest_val = float(hist[-1].get("sumOpenInterestValue") or hist[-1].get("sumOpenInterest") or 0.0)
        if oldest_val == 0.0:
            return None
        return ((latest_val - oldest_val) / oldest_val) * 100.0
    except Exception as e:
        print(f"[BINANCE OI 1H ERROR] {symbol}: {e}")
        return None


class OIHistoryTracker:
    """
    Класс для накопления и отслеживания истории Open Interest (открытого интереса)
    в оперативной памяти с целью расчёта процентного изменения за период.
    """
    def __init__(self, window_candles: int, timeframe: str):
        self.window_candles: int = window_candles
        self.timeframe: str = timeframe
        # Структура: { symbol: [(timestamp, oi_value), ...] }
        self.history: Dict[str, List[Tuple[float, float]]] = {}
        
    def add_entry(self, symbol: str, oi: float) -> None:
        """Добавляет новое измерение OI в историю и очищает старые записи."""
        if symbol not in self.history:
            self.history[symbol] = []
        now = time.time()
        self.history[symbol].append((now, oi))
        
        # Рассчитываем максимальное время хранения данных:
        # window_candles * минут в свече * 60 сек, добавляем 50% буфера
        timeframe_mins = get_timeframe_minutes(self.timeframe)
        max_age_seconds = self.window_candles * timeframe_mins * 60
        # Обеспечиваем удержание истории минимум на 2 часа (7200 секунд) для контроля OI за последний час
        max_age_seconds = max(max_age_seconds, 7200)
        cutoff = now - max_age_seconds * 1.5
        
        # Очищаем историю от слишком старых записей для экономии памяти
        self.history[symbol] = [entry for entry in self.history[symbol] if entry[0] > cutoff]
        
    def get_percentage_change(self, symbol: str, current_oi: float) -> Tuple[Optional[float], str]:
        """
        Рассчитывает процент изменения OI относительно значения N свечей назад.
        
        Returns:
            Tuple[Optional[float], str]: Процент изменения (или None) и статус-сообщение.
        """
        if symbol not in self.history or len(self.history[symbol]) < 2:
            return None, "Накапливается история (нужно минимум 2 сканирования)"
            
        timeframe_mins = get_timeframe_minutes(self.timeframe)
        target_age_seconds = self.window_candles * timeframe_mins * 60
        now = time.time()
        target_time = now - target_age_seconds
        
        history = self.history[symbol]
        # Находим запись, наиболее близкую к целевому времени
        closest_entry = min(history, key=lambda x: abs(x[0] - target_time))
        
        # Если ближайшая запись слишком свежая (меньше 80% от целевого времени),
        # мы возвращаем сравнение с самой старой имеющейся записью, но помечаем статус.
        if (now - closest_entry[0]) < target_age_seconds * 0.8:
            oldest_entry = history[0]
            actual_age_mins = (now - oldest_entry[0]) / 60.0
            historical_oi = oldest_entry[1]
            if historical_oi == 0:
                return 0.0, f"Накапливается история ({actual_age_mins:.1f} мин.)"
            change = ((current_oi - historical_oi) / historical_oi) * 100.0
            return change, f"Накапливается история ({actual_age_mins:.1f} мин. из {target_age_seconds/60:.1f} мин.)"
        else:
            historical_oi = closest_entry[1]
            if historical_oi == 0:
                return 0.0, "Базовое значение OI = 0"
            change = ((current_oi - historical_oi) / historical_oi) * 100.0
            return change, "История полностью накоплена"

    def get_hourly_change(self, symbol: str, current_oi: float) -> Optional[float]:
        """
        Рассчитывает процент изменения OI за последний час.
        Возвращает процент изменения (например, -0.6 для падения на 0.6%) или None, если истории недостаточно.
        """
        if symbol not in self.history or len(self.history[symbol]) < 2:
            return None
            
        now = time.time()
        target_time = now - 3600  # 1 час назад
        
        history = self.history[symbol]
        # Находим запись, наиболее близкую к целевому времени (1 час назад)
        closest_entry = min(history, key=lambda x: abs(x[0] - target_time))
        
        # Если ближайшая запись старше 15 минут от целевого времени,
        # значит у нас нет достаточно старой истории за этот час
        if abs(closest_entry[0] - target_time) > 900:  # 15 минут
            return None
            
        old_oi = closest_entry[1]
        if old_oi == 0:
            return 0.0
        return ((current_oi - old_oi) / old_oi) * 100.0


def calculate_indicators(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Расчёт экспоненциальных скользящих средних (EMA) и гистограммы MACD с использованием pandas.
    
    Args:
        df (pd.DataFrame): DataFrame со свечными данными (должна содержать колонку 'close').
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (ema_fast, ema_slow, macd_hist)
    """
    # Расчёт EMA
    ema_fast = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # Расчёт MACD
    macd_line = df['close'].ewm(span=MACD_FAST, adjust=False).mean() - df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    return ema_fast, ema_slow, macd_hist


def check_technical_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Анализирует пересечения EMA и гистограммы MACD на завершённых свечах.
    
    Для исключения шумов вычисления проводятся на закрытых свечах:
      - Индекс -2 (последняя закрытая свеча)
      - Индекс -3 (предыдущая закрытая свеча)
      - Индекс -4 (свеча перед ней, для поиска пересечений с лагом в 1 свечу)
      
    Args:
        df (pd.DataFrame): Свечные данные от API.
        
    Returns:
        Dict[str, Any]: Результаты анализа (signal, ema_fast_val, ema_slow_val, macd_hist_val, details).
    """
    result = {
        "signal": "NEUTRAL",
        "ema_fast_val": 0.0,
        "ema_slow_val": 0.0,
        "macd_hist_val": 0.0,
        "details": ""
    }
    
    min_required_len = max(EMA_SLOW, MACD_SLOW + MACD_SIGNAL) + 5
    if len(df) < min_required_len:
        result["details"] = f"Недостаточно свечей ({len(df)} < {min_required_len})"
        return result
        
    ema_fast, ema_slow, macd_hist = calculate_indicators(df)
    
    # Текущие закрытые значения (индекс -2)
    val_fast_curr = float(ema_fast.iloc[-2])
    val_slow_curr = float(ema_slow.iloc[-2])
    hist_curr = float(macd_hist.iloc[-2])
    
    # Предыдущие закрытые значения (индекс -3)
    val_fast_prev = float(ema_fast.iloc[-3])
    val_slow_prev = float(ema_slow.iloc[-3])
    hist_prev = float(macd_hist.iloc[-3])
    
    # Значения позапрошлые (индекс -4) для поиска пересечений с небольшим лагом
    val_fast_prev2 = float(ema_fast.iloc[-4])
    val_slow_prev2 = float(ema_slow.iloc[-4])
    hist_prev2 = float(macd_hist.iloc[-4])
    
    result["ema_fast_val"] = val_fast_curr
    result["ema_slow_val"] = val_slow_curr
    result["macd_hist_val"] = hist_curr
    
    # Бычий/медвежий статус на последней закрытой свече
    ema_bullish = val_fast_curr > val_slow_curr
    ema_bearish = val_fast_curr < val_slow_curr
    macd_bullish = hist_curr > 0
    macd_bearish = hist_curr < 0
    
    # 1. Пересечение EMA
    ema_cross_up = (val_fast_prev <= val_slow_prev) and (val_fast_curr > val_slow_curr)
    ema_cross_down = (val_fast_prev >= val_slow_prev) and (val_fast_curr < val_slow_curr)
    # С лагом в 1 свечу
    ema_cross_up_lag = (val_fast_prev2 <= val_slow_prev2) and (val_fast_prev > val_slow_prev)
    ema_cross_down_lag = (val_fast_prev2 >= val_slow_prev2) and (val_fast_prev < val_slow_prev)
    
    # 2. Пересечение MACD через ноль
    macd_cross_up = (hist_prev <= 0.0) and (hist_curr > 0.0)
    macd_cross_down = (hist_prev >= 0.0) and (hist_curr < 0.0)
    # С лагом в 1 свечу
    macd_cross_up_lag = (hist_prev2 <= 0.0) and (hist_prev > 0.0)
    macd_cross_down_lag = (hist_prev2 >= 0.0) and (hist_prev < 0.0)
    
    # Сигнал формируется строго на основе пересечения EMA и согласования тренда
    is_buy = ema_bullish and (ema_cross_up or ema_cross_up_lag)
    is_sell = ema_bearish and (ema_cross_down or ema_cross_down_lag)
    
    if is_buy:
        result["signal"] = "BUY"
        cross_desc = []
        if ema_cross_up or ema_cross_up_lag:
            cross_desc.append("пересечение EMA вверх")
        result["details"] = " и ".join(cross_desc) if cross_desc else "восходящий тренд EMA"
    elif is_sell:
        result["signal"] = "SELL"
        cross_desc = []
        if ema_cross_down or ema_cross_down_lag:
            cross_desc.append("пересечение EMA вниз")
        result["details"] = " и ".join(cross_desc) if cross_desc else "нисходящий тренд EMA"
    else:
        result["signal"] = "NEUTRAL"
        result["details"] = f"Нет сигнала (EMA: {'Бычий' if ema_bullish else 'Медвежий'})"
        
    return result


def generate_chart(symbol: str, df_klines: pd.DataFrame) -> Optional[str]:
    """
    Генерирует свечной график с EMA и гистограммой MACD, сохраняет во временный файл.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Считаем индикаторы для всего датафрейма (чтобы избежать краевых эффектов)
        ema_fast, ema_slow, macd_hist = calculate_indicators(df_klines)
        
        # Для визуализации берем последние 45 свечей
        tail_size = 45
        df = df_klines.tail(tail_size).copy()
        ema_f = ema_fast.tail(tail_size)
        ema_s = ema_slow.tail(tail_size)
        m_hist = macd_hist.tail(tail_size)
        
        # Создаем фигуру с двумя сабплотами (верхний для цены, нижний для MACD)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
        fig.patch.set_facecolor('#121212') # Sleek dark mode background
        
        ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('#1a1a1a')
        
        # Индексы для оси X
        x = np.arange(len(df))
        
        # Отрисовка свечей
        for i in range(len(df)):
            row = df.iloc[i]
            # Свеча растет (зеленая) или падает (красная)
            color = '#00c087' if row['close'] >= row['open'] else '#ff3b30'
            
            # Фитиль (High-Low)
            ax1.vlines(x[i], row['low'], row['high'], color=color, linewidth=1.5)
            # Тело (Open-Close)
            body_bottom = min(row['open'], row['close'])
            body_height = abs(row['close'] - row['open'])
            if body_height == 0:
                body_height = (row['high'] - row['low']) * 0.05
            
            ax1.bar(x[i], body_height, bottom=body_bottom, color=color, width=0.6, align='center', edgecolor=color, linewidth=0.5)
            
        # Рисуем линии EMA
        ax1.plot(x, ema_f, color='#2f80ed', linewidth=1.5, label=f'EMA({EMA_FAST})')
        ax1.plot(x, ema_s, color='#f2994a', linewidth=1.5, label=f'EMA({EMA_SLOW})')
        
        ax1.set_title(f"{symbol} ({TIMEFRAME}) - Сигнальный график", color='white', fontsize=14, fontweight='bold', pad=10)
        ax1.grid(True, color='#2c2c2c', linestyle='--', alpha=0.5)
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#2c2c2c', labelcolor='white')
        
        # Отрисовка MACD гистограммы
        for i in range(len(m_hist)):
            val = m_hist.iloc[i]
            color = '#00c087' if val >= 0 else '#ff3b30'
            ax2.bar(x[i], val, color=color, width=0.6, align='center')
            
        ax2.set_title("MACD гистограмма", color='white', fontsize=10, pad=5)
        ax2.grid(True, color='#2c2c2c', linestyle='--', alpha=0.5)
        ax2.tick_params(colors='white')
        
        # Настройка временных меток на оси X
        step = max(1, len(df) // 8)
        xticks = x[::step]
        xticklabels = [df['time'].iloc[i].strftime('%d.%m %H:%M') for i in xticks]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticklabels, rotation=30, ha='right', color='white')
        
        plt.tight_layout()
        
        # Папка для сохранения временных файлов
        os.makedirs("scratch", exist_ok=True)
        chart_path = f"scratch/{symbol.replace('/', '_')}_chart.png"
        plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
        plt.close()
        
        return chart_path
    except Exception as e:
        print(f"[CHART ERROR] Ошибка генерации свечного графика для {symbol}: {e}")
        return None


def send_telegram_signal(symbol: str, direction: str, mark_price: float, oi: float, oi_status: str,
                         funding_rate: float, ema_fast_val: float, ema_slow_val: float, macd_val: float,
                         analysis_details: str, df_klines: pd.DataFrame) -> None:
    """
    Формирует красивое HTML-сообщение в Telegram, генерирует свечной график
    и отправляет фото с описанием в Telegram.
    """
    bot_token = BOT_TOKEN
    chat_id = CHAT_ID
    
    if not bot_token or not chat_id:
        print("[TG] Переменные TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы. Пропуск отправки в Telegram.")
        return

    # 1. Генерация графика
    chart_path = generate_chart(symbol, df_klines)
    
    # 2. Формирование сообщения
    emoji = "🟢" if direction == "BUY" else "🔴"
    dir_ru = "ЛОНГ (BUY)" if direction == "BUY" else "ШОРТ (SELL)"
    funding_pct = funding_rate * 100
    
    # Красивая верстка в HTML
    text = (
        f"{emoji} <b>СИГНАЛ: {dir_ru} для {symbol}</b>\n\n"
        f"💵 <b>Цена маркировки:</b> <code>{mark_price:.5f}</code>\n"
        f"📊 <b>Open Interest:</b> <code>{oi:,.2f} USDT</code> ({oi_status})\n"
        f"💸 <b>Funding Rate:</b> <code>{funding_pct:+.4f}%</code> (8h)\n\n"
        f"📈 <b>EMA({EMA_FAST}) &gt; EMA({EMA_SLOW}):</b> <code>{ema_fast_val:.5f}</code> {'&gt;' if ema_fast_val > ema_slow_val else '&lt;'} <code>{ema_slow_val:.5f}</code>\n"
        f"📉 <b>MACD гистограмма:</b> <code>{macd_val:.5f}</code> ({analysis_details})\n\n"
        f"🕒 <i>Время сигнала: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    )
    
    # 3. Отправка фото в Telegram
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    try:
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                payload = {
                    "chat_id": chat_id,
                    "caption": text,
                    "parse_mode": "HTML"
                }
                files = {"photo": photo}
                r = requests.post(url, data=payload, files=files, timeout=15)
            # Удаляем временный файл графика
            try:
                os.remove(chart_path)
            except Exception:
                pass
        else:
            # Если график не сгенерировался, шлем обычным текстом
            url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            r = requests.post(url_msg, data=payload, timeout=15)
            
        if r.status_code == 200:
            print(f"[TG] Сигнал {direction} по {symbol} успешно отправлен в Telegram!")
        else:
            print(f"[TG] Ошибка отправки в Telegram: Код {r.status_code}, Ответ: {r.text}")
    except Exception as e:
        print(f"[TG] Исключение при отправке в Telegram: {e}")


# Глобальный кэш параметров контрактов
CONTRACT_PARAMS: Dict[str, Dict[str, Any]] = {}

def load_contract_params(client: BingxClient) -> None:
    """Загружает параметры контрактов (точность цены и количества) с биржи."""
    global CONTRACT_PARAMS
    try:
        url = f"{client.BASE_URL}/openApi/swap/v2/quote/contracts"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == 0 and "data" in data:
                for item in data["data"]:
                    sym = item.get("symbol")
                    if sym:
                        CONTRACT_PARAMS[sym] = {
                            "qty_precision": int(item.get("quantityPrecision", 4)),
                            "price_precision": int(item.get("pricePrecision", 4)),
                            "max_leverage": int(item.get("maxLongLeverage", 20))
                        }
                print(f"[INFO] Успешно загружены параметры для {len(CONTRACT_PARAMS)} контрактов.")
            else:
                print(f"[WARNING] Не удалось загрузить параметры контрактов. Код биржи: {data.get('code')}")
        else:
            print(f"[WARNING] Не удалось загрузить параметры контрактов. Статус: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки параметров контрактов: {e}")

def round_qty(symbol: str, qty: float) -> float:
    """Округляет количество до шага, разрешенного биржей."""
    params = CONTRACT_PARAMS.get(symbol, {})
    precision = params.get("qty_precision", 4)
    return round(qty, precision)

def round_price(symbol: str, price: float) -> float:
    """Округляет цену до шага, разрешенного биржей."""
    params = CONTRACT_PARAMS.get(symbol, {})
    precision = params.get("price_precision", 4)
    return round(price, precision)

def log_exchange_error(action: str, symbol: str, response: Dict[str, Any]) -> None:
    """
    Логирует ошибки биржи (code != 0) и отправляет подробное уведомление в Telegram.
    """
    code = response.get("code")
    msg = response.get("msg", "Нет сообщения об ошибке")
    
    err_text = (
        f"⚠️ <b>ОШИБКА БИРЖИ BingX</b>\n\n"
        f"💻 <b>Действие:</b> <code>{action}</code>\n"
        f"🪙 <b>Актив:</b> <code>{symbol}</code>\n"
        f"🚫 <b>Код ошибки:</b> <code>{code}</code>\n"
        f"📝 <b>Сообщение:</b> <code>{msg}</code>\n\n"
        f"📌 <i>Пожалуйста, проверьте баланс, плечо, лимиты ордера или доступность пары на бирже.</i>"
    )
    
    print(f"[EXCHANGE ERROR] {action} {symbol} failed! Code: {code}, Msg: {msg}")
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": err_text,
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            print(f"[TG ERROR] Не удалось отправить ошибку в Telegram: {e}")

def log_trade_success(action: str, symbol: str, side: str, qty: float, price: float, tp: float, sl: float) -> None:
    """
    Отправляет уведомление об успешном открытии сделки в Telegram.
    """
    emoji = "🚀" if side == "long" else "📉"
    side_str = "ДЛИННАЯ (LONG)" if side == "long" else "КОРОТКАЯ (SHORT)"
    
    text = (
        f"{emoji} <b>УСПЕШНАЯ СДЕЛКА: {action}</b>\n\n"
        f"🪙 <b>Актив:</b> <code>{symbol}</code>\n"
        f"↕️ <b>Направление:</b> <code>{side_str}</code>\n"
        f"📦 <b>Объем:</b> <code>{qty} {symbol.split('-')[0]}</code>\n"
        f"💵 <b>Цена входа:</b> <code>{price:.5f}</code>\n"
        f"🎯 <b>Take-Profit:</b> <code>{tp:.5f}</code>\n"
        f"🛑 <b>Stop-Loss:</b> <code>{sl:.5f}</code>\n\n"
        f"⚡ <i>Ордера на Тейк-Профит, Стоп-Лосс и Трейлинг успешно выставлены.</i>"
    )
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            print(f"[TG ERROR] {e}")

def log_position_closed(reason: str, symbol: str, side: str, qty: float, metric_val: str) -> None:
    """
    Отправляет уведомление в Telegram о принудительном закрытии позиции мониторингом.
    """
    side_str = "ДЛИННАЯ (LONG)" if side == "long" else "КОРОТКАЯ (SHORT)"
    text = (
        f"🔴 <b>ПОЗИЦИЯ ЗАКРЫТА МОНИТОРИНГОМ</b>\n\n"
        f"🪙 <b>Актив:</b> <code>{symbol}</code>\n"
        f"↕️ <b>Направление:</b> <code>{side_str}</code>\n"
        f"📦 <b>Объем:</b> <code>{qty} {symbol.split('-')[0]}</code>\n"
        f"⚠️ <b>Причина:</b> {reason}\n"
        f"📊 <b>Метрика:</b> <code>{metric_val}</code>\n\n"
        f"🛑 <i>Все связанные отложенные ордера успешно отменены.</i>"
    )
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            print(f"[TG ERROR] {e}")

def close_position(client: BingxClient, symbol: str, side: str, qty: float, one_way_mode: bool = False) -> Dict[str, Any]:
    """
    Закрывает открытую позицию рыночным ордером в противоположном направлении.
    """
    s = client._to_bingx_symbol(symbol)
    
    # Сначала отменяем все связанные лимитные ордера TP/SL и трейлинг
    try:
        client.cancel_existing_orders(symbol)
    except Exception as e:
        print(f"[ERROR] Ошибка отмены ордеров перед закрытием для {symbol}: {e}")
        
    pos_side = "BOTH" if one_way_mode else ("LONG" if side == "long" else "SHORT")
    order_side = "SELL" if side == "long" else "BUY"
    
    params = {
        "symbol": s,
        "side": order_side,
        "positionSide": pos_side,
        "type": "MARKET",
        "quantity": qty,
        "recvWindow": 5000,
        "timestamp": int(time.time() * 1000)
    }
    
    if one_way_mode:
        params["reduceOnly"] = "true"
        
    return client._request("POST", "/openApi/swap/v2/trade/order", params)


def main(client: BingxClient) -> None:
    """
    Основной бесконечный цикл сканера фьючерсного рынка BingX.
    
    Шаги работы:
      1. Запрашивает список всех активных контрактов.
      2. Делает групповой (bulk) запрос цен и ставок финансирования всех токенов.
      3. Применяет фильтры Open Interest и Funding Rate.
      4. По прошедшим фильтр токенам запрашивает историю свечей.
      5. Проводит технический анализ (EMA, MACD).
      6. При обнаружении сигналов выводит структурированное уведомление в консоль.
    """
    print("======================================================================")
    print("🚀 ЗАПУСК ТОРГОВОЙ СИСТЕМЫ СКАНИРОВАНИЯ BINGX")
    print(f"🕒 Время старта: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("----------------------------------------------------------------------")
    print(f"📊 Настройки индикаторов: Таймфрейм = {TIMEFRAME}, Свечей для расчёта = {KLINES_LIMIT}")
    print(f"📈 Параметры EMA: Fast = {EMA_FAST}, Slow = {EMA_SLOW}")
    print(f"📉 Параметры MACD: Fast = {MACD_FAST}, Slow = {MACD_SLOW}, Signal = {MACD_SIGNAL}")
    print(f"⚙️ Режим фильтрации: {FILTER_MODE}")
    print(f"💰 Порог ставки финансирования: {FUNDING_RATE_THRESHOLD * 100:.4f}%")
    if USE_OI_CHANGE:
        print(f"⏳ Порог изменения Open Interest: >= {OI_CHANGE_PERCENT}% (за последние 24 часа с Binance)")
    else:
        print(f"💎 Порог Open Interest: >= {OI_THRESHOLD:,} USDT (абсолютное значение)")
    print("======================================================================\n")
    
    # Загружаем параметры контрактов с биржи при старте сканера
    load_contract_params(client)
    
    oi_tracker = OIHistoryTracker(window_candles=OI_CHANGE_WINDOW, timeframe=TIMEFRAME)
    signal_cooldowns: Dict[Tuple[str, str], float] = {}
    last_monitoring_time: float = 0.0
    
    while True:
        try:
            start_time = time.time()
            scan_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{scan_time_str}] Запуск нового цикла сканирования...")
            
            # --- ПЛАНОВЫЙ МОНИТОРИНГ ПОЗИЦИЙ (РАЗ В 5 МИНУТ) ---
            now = time.time()
            if now - last_monitoring_time >= MONITORING_INTERVAL_SECONDS:
                last_monitoring_time = now
                print(f"[{scan_time_str}] Запуск планового мониторинга позиций (раз в 5 минут)...")
                try:
                    positions = client.get_positions()
                    active_positions_count = 0
                    for pos in positions:
                        qty = float(pos.get("positionAmt", 0.0))
                        if abs(qty) > 0:
                            active_positions_count += 1
                            symbol = pos.get("symbol")
                            pos_side = pos.get("positionSide")
                            
                            # Переводим в нормальный формат с дефисом
                            norm_symbol = symbol.replace("USDT", "-USDT")
                            
                            # Определяем сторону позиции: long/short
                            side = "long" if pos_side in ("LONG", "BOTH") and qty > 0 else "short"
                            
                            print(f"  [MONITOR] Проверка позиции {pos_side} по {norm_symbol} (размер: {qty})...")
                            
                            # 1. Проверяем ставку финансирования
                            funding_rate = client.get_funding_rate(norm_symbol)
                            if funding_rate is not None:
                                print(f"    Фандинг {norm_symbol}: {funding_rate*100:+.6f}% (Лимиты: [{funding_min*100}%, {funding_max*100}%])")
                                if (funding_rate < funding_min and side == 'short') or (funding_rate > funding_max and side == 'long'):
                                    print(f"    [CLOSE] Фандинг {funding_rate*100:+.4f}% вышел за допустимые пределы!")
                                    close_resp = close_position(client, norm_symbol, side, abs(qty), ONE_WAY_MODE)
                                    if close_resp.get("code") != 0:
                                        log_exchange_error("Закрытие позиции по фандингу", norm_symbol, close_resp)
                                    else:
                                        log_position_closed(
                                            reason="выход ставки финансирования за лимиты",
                                            symbol=norm_symbol,
                                            side=side,
                                            qty=abs(qty),
                                            metric_val=f"Funding = {funding_rate*100:+.4f}%"
                                        )
                                    continue
                            
                            # 2. Проверяем падение открытого интереса за последний час с Binance
                            hourly_oi_change = get_binance_oi_change_1h(norm_symbol)
                            if hourly_oi_change is not None:
                                print(f"    Изменение OI за час для {norm_symbol} (Binance): {hourly_oi_change:+.2f}% (Порог: -0.5%)")
                                if hourly_oi_change < -0.5:
                                    print(f"    [CLOSE] Открытый интерес упал за час на {hourly_oi_change:.2f}% (более чем на 0.5%)!")
                                    close_resp = close_position(client, norm_symbol, side, abs(qty), ONE_WAY_MODE)
                                    if close_resp.get("code") != 0:
                                        log_exchange_error("Закрытие позиции по падению OI", norm_symbol, close_resp)
                                    else:
                                        log_position_closed(
                                            reason="падение открытого интереса за час более чем на 0.5%",
                                            symbol=norm_symbol,
                                            side=side,
                                            qty=abs(qty),
                                            metric_val=f"Изменение OI = {hourly_oi_change:+.2f}%"
                                        )
                                    continue
                            else:
                                print(f"    [MONITOR] Не удалось получить данные часового изменения OI для {norm_symbol} с Binance.")
                    print(f"[{scan_time_str}] Мониторинг завершен. Активных позиций проверено: {active_positions_count}.")
                except Exception as mon_ex:
                    print(f"[{scan_time_str}] [ERROR] Исключение во время мониторинга позиций: {mon_ex}")
            
            # 1. Получаем список активных фьючерсных токенов
            tickers = client.get_all_tikers()
            if not tickers:
                print(f"[{scan_time_str}] [WARNING] Не удалось получить список торговых пар от BingX. Повтор через {SCAN_INTERVAL_SECONDS}с.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue
            
            ticker_set = set(tickers)
            
            # 2. Оптимизированный групповой (bulk) запрос цен и финансирования
            premium_indices = client.get_premium_index()
            if not premium_indices:
                print(f"[{scan_time_str}] [WARNING] Не удалось получить премиум-индексы. Повтор через {SCAN_INTERVAL_SECONDS}с.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue
                
            # Группируем полученные данные по символам
            market_data: Dict[str, Dict[str, float]] = {}
            for item in premium_indices:
                symbol = item.get("symbol")
                if symbol in ticker_set:
                    try:
                        market_data[symbol] = {
                            "funding_rate": float(item.get("lastFundingRate") or 0.0),
                            "mark_price": float(item.get("markPrice") or 0.0),
                            "index_price": float(item.get("indexPrice") or 0.0)
                        }
                    except (ValueError, TypeError):
                        continue
                        
            print(f"[{scan_time_str}] Получены рыночные данные для {len(market_data)} фьючерсных пар.")
            
            # 3. Фильтрация по Funding Rate
            passed_funding: List[str] = []
            for symbol, data in market_data.items():
                if abs(data["funding_rate"]) >= FUNDING_RATE_THRESHOLD:
                    passed_funding.append(symbol)
                    
            print(f"[{scan_time_str}] Фильтр ставки финансирования (>= {FUNDING_RATE_THRESHOLD*100:.4f}%) прошли {len(passed_funding)} пар.")
            
            # 4. Проверка фильтра Open Interest и составление списка кандидатов
            candidates: List[Tuple[str, float, str]] = []
            
            if FILTER_MODE == "AND":
                # В режиме AND запрашиваем OI только для тех, кто прошёл Funding Rate фильтр
                # Это минимизирует количество запросов и исключает rate limit
                for symbol in passed_funding:
                    try:
                        time.sleep(0.2)  # Безопасная задержка для API
                        oi = client.get_open_insterest(symbol)
                        if oi is None:
                            continue
                            
                        # Сохраняем в локальный трекер для совместимости
                        oi_tracker.add_entry(symbol, oi)
                        
                        oi_passed = False
                        oi_status_msg = ""
                        
                        if USE_OI_CHANGE:
                            oi_change = get_binance_oi_change_24h(symbol)
                            if oi_change is not None:
                                if abs(oi_change) >= OI_CHANGE_PERCENT:
                                    oi_passed = True
                                    oi_status_msg = f"Binance 24h OI: {oi_change:+.2f}%"
                                else:
                                    oi_status_msg = f"Binance 24h OI: {oi_change:+.2f}% (недостаточно)"
                            else:
                                if oi >= OI_THRESHOLD:
                                    oi_passed = True
                                    oi_status_msg = f"Сбой Binance (резервный абсолютный OI {oi:,.0f} >= {OI_THRESHOLD:,.0f})"
                                else:
                                    oi_status_msg = "Сбой Binance (не пройден резервный порог OI)"
                        else:
                            if oi >= OI_THRESHOLD:
                                oi_passed = True
                                oi_status_msg = "Абсолютный порог"
                                
                        if oi_passed:
                            candidates.append((symbol, oi, oi_status_msg))
                    except Exception as e:
                        print(f"[{scan_time_str}] [ERROR] Ошибка при проверке OI для {symbol}: {e}")
                        continue
            
            else:  # FILTER_MODE == "OR"
                # В режиме OR нам нужно проверять все пары на OI, что может вызвать задержки
                print(f"[{scan_time_str}] [WARNING] Внимание: режим OR опрашивает все {len(market_data)} пар. Это может занять до 2-3 минут.")
                for symbol in market_data.keys():
                    try:
                        # Если пара уже прошла Funding Rate, она добавляется в кандидаты автоматически
                        if symbol in passed_funding:
                            time.sleep(0.1)
                            oi = client.get_open_insterest(symbol) or 0.0
                            oi_tracker.add_entry(symbol, oi)
                            candidates.append((symbol, oi, "Прошёл по Funding Rate"))
                            continue
                            
                        # Иначе опрашиваем и проверяем OI
                        time.sleep(0.1)
                        oi = client.get_open_insterest(symbol)
                        if oi is None:
                            continue
                            
                        oi_tracker.add_entry(symbol, oi)
                        
                        oi_passed = False
                        oi_status_msg = ""
                        
                        if USE_OI_CHANGE:
                            oi_change = get_binance_oi_change_24h(symbol)
                            if oi_change is not None:
                                if abs(oi_change) >= OI_CHANGE_PERCENT:
                                    oi_passed = True
                                    oi_status_msg = f"Binance 24h OI: {oi_change:+.2f}%"
                                else:
                                    oi_status_msg = f"Binance 24h OI: {oi_change:+.2f}% (недостаточно)"
                            else:
                                if oi >= OI_THRESHOLD:
                                    oi_passed = True
                                    oi_status_msg = f"Сбой Binance (резервный абсолютный OI {oi:,.0f} >= {OI_THRESHOLD:,.0f})"
                                else:
                                    oi_status_msg = "Сбой Binance (не пройден резервный порог OI)"
                        else:
                            if oi >= OI_THRESHOLD:
                                oi_passed = True
                                oi_status_msg = "Абсолютный порог"
                                
                        if oi_passed:
                            candidates.append((symbol, oi, oi_status_msg))
                    except Exception as e:
                        continue
                        
            print(f"[{scan_time_str}] Первичную фильтрацию ({FILTER_MODE}) успешно прошли {len(candidates)} пар.")
            
            # 5. Детальный технический анализ кандидатов
            for symbol, oi, oi_status in candidates:
                try:
                    time.sleep(0.2)  # Безопасный интервал между запросами свечей
                    
                    df_klines = client.get_klines(symbol, TIMEFRAME, limit=KLINES_LIMIT)
                    if df_klines.empty:
                        continue
                        
                    analysis = check_technical_signals(df_klines)
                    
                    if analysis["signal"] in ("BUY", "SELL"):
                        mark_price = market_data[symbol]["mark_price"]
                        funding_rate = market_data[symbol]["funding_rate"]
                        
                        # Вывод оформленного сигнала в консоль
                        direction_str = "BUY (LONG)" if analysis["signal"] == "BUY" else "SELL (SHORT)"
                        funding_pct_str = f"{funding_rate * 100:+.4f}%"
                        
                        ema_fast_val = analysis["ema_fast_val"]
                        ema_slow_val = analysis["ema_slow_val"]
                        macd_val = analysis["macd_hist_val"]
                        
                        print("\n==================================================")
                        print(f"[{scan_time_str}] СИГНАЛ: {direction_str} для {symbol}")
                        print(f"   Цена маркировки: {mark_price:.5f}")
                        print(f"   Open Interest: {oi:,.2f} USDT ({oi_status})")
                        print(f"   Funding Rate: {funding_pct_str} (8h)")
                        print(f"   EMA({EMA_FAST}) > EMA({EMA_SLOW}): {ema_fast_val:.5f} {'/ > /' if ema_fast_val > ema_slow_val else '/ < /'} {ema_slow_val:.5f}")
                        print(f"   MACD гистограмма: {macd_val:.5f} ({analysis['details']})")
                        print("==================================================")
                        
                        # --- Отправка в Telegram с контролем кулдауна ---
                        sig_key = (symbol, analysis["signal"])
                        now = time.time()
                        last_sent = signal_cooldowns.get(sig_key, 0.0)
                        if (now - last_sent) >= (TELEGRAM_COOLDOWN_HOURS * 3600):
                            signal_cooldowns[sig_key] = now
                            send_telegram_signal(
                                symbol=symbol,
                                direction=analysis["signal"],
                                mark_price=mark_price,
                                oi=oi,
                                oi_status=oi_status,
                                funding_rate=funding_rate,
                                ema_fast_val=ema_fast_val,
                                ema_slow_val=ema_slow_val,
                                macd_val=macd_val,
                                analysis_details=analysis["details"],
                                df_klines=df_klines
                            )
                            
                            # --- ИСПОЛНЕНИЕ СДЕЛКИ НА БИРЖЕ ---
                            if TRADE_ENABLED:
                                try:
                                    side = "long" if analysis["signal"] == "BUY" else "short"
                                    
                                    # Проверяем направление ставки финансирования перед входом
                                    if side == "long" and funding_rate > funding_max:
                                        print(f"[TRADE] Пропуск ЛОНГА по {symbol}: фандинг {funding_rate*100:+.6f}% выше порога {funding_max*100:+.6f}%")
                                        continue
                                    if side == "short" and funding_rate < funding_min:
                                        print(f"[TRADE] Пропуск ШОРТА по {symbol}: фандинг {funding_rate*100:+.6f}% ниже порога {funding_min*100:+.6f}%")
                                        continue
                                        
                                    # Проверяем, есть ли уже открытая позиция по этому символу
                                    existing_pos = client.get_open_position(symbol, side)
                                    if existing_pos:
                                        print(f"[TRADE] Позиция {side.upper()} по {symbol} уже существует. Пропускаем открытие.")
                                    else:
                                        print(f"[TRADE] Открываем позицию {side.upper()} по {symbol}...")
                                        
                                        # 1. Устанавливаем кредитное плечо
                                        lev_resp = client.set_leverage_bx(symbol, side, LEVERAGE, ONE_WAY_MODE)
                                        if lev_resp.get("code") != 0:
                                            log_exchange_error("Установка плеча", symbol, lev_resp)
                                            
                                        # 2. Рассчитываем объем сделки с учетом точности контракта
                                        contract_val = TRADE_SIZE_USDT * LEVERAGE
                                        raw_qty = contract_val / mark_price
                                        qty = round_qty(symbol, raw_qty)
                                        
                                        if qty <= 0:
                                            print(f"[TRADE ERROR] Объем сделки {qty} слишком мал после округления. Пропуск сделки.")
                                            continue
                                            
                                        # 3. Рассчитываем цены для TP и SL
                                        if side == "long":
                                            tp_price = round_price(symbol, mark_price * (1 + TP_PERCENT / 100))
                                            sl_price = round_price(symbol, mark_price * (1 - SL_PERCENT / 100))
                                            trailing_act = round_price(symbol, mark_price * (1 + TRAILING_ACTIVATION_PERCENT / 100))
                                        else:
                                            tp_price = round_price(symbol, mark_price * (1 - TP_PERCENT / 100))
                                            sl_price = round_price(symbol, mark_price * (1 + SL_PERCENT / 100))
                                            trailing_act = round_price(symbol, mark_price * (1 - TRAILING_ACTIVATION_PERCENT / 100))
                                            
                                        # 4. Открываем маркет ордер со встроенным TP/SL
                                        order_resp = client.place_market_order(
                                            side=side,
                                            qty=qty,
                                            symbol=symbol,
                                            stop=sl_price,
                                            tp=tp_price,
                                            pos_side_BOTH=ONE_WAY_MODE
                                        )
                                        
                                        if order_resp.get("code") != 0:
                                            log_exchange_error("Открытие рыночного ордера", symbol, order_resp)
                                        else:
                                            # Получаем точную среднюю цену входа
                                            order_data = order_resp.get("data", {}).get("order", {})
                                            entry_price = float(order_data.get("avgPrice") or order_data.get("price") or mark_price)
                                            if entry_price == 0:
                                                entry_price = mark_price
                                                
                                            print(f"[TRADE SUCCESS] Успешно открыта позиция {side.upper()} по {symbol} по цене {entry_price:.5f}!")
                                            log_trade_success("ОТКРЫТИЕ ПОЗИЦИИ", symbol, side, qty, entry_price, tp_price, sl_price)
                                            
                                            # 5. Постановка трейлинга
                                            time.sleep(0.5)
                                            trail_rate = TRAILING_RATE_PERCENT / 100
                                            trail_resp = client.set_trailing(
                                                symbol=client._to_bingx_symbol(symbol),
                                                side=side,
                                                qty=qty,
                                                activation_price=trailing_act,
                                                priceRate=trail_rate,
                                                BOTH=ONE_WAY_MODE
                                            )
                                            
                                            if trail_resp.get("code") != 0:
                                                log_exchange_error("Постановка трейлинг ордера", symbol, trail_resp)
                                            else:
                                                print(f"[TRADE SUCCESS] Трейлинг-стоп ордер успешно выставлен для {symbol}!")
                                except Exception as trade_ex:
                                    print(f"[TRADE EXCEPTION] Критическая ошибка в модуле торговли для {symbol}: {trade_ex}")
                        else:
                            cooldown_left_sec = (TELEGRAM_COOLDOWN_HOURS * 3600) - (now - last_sent)
                            print(f"[TG INFO] Сигнал {analysis['signal']} по {symbol} пропущен (кулдаун, осталось {cooldown_left_sec / 3600:.2f} ч.)")
                        
                except Exception as ex:
                    print(f"[{scan_time_str}] [ERROR] Не удалось рассчитать индикаторы для {symbol}: {ex}")
                    continue
            
            # Расчёт времени сна
            elapsed = time.time() - start_time
            sleep_time = max(1.0, SCAN_INTERVAL_SECONDS - elapsed)
            print(f"[{scan_time_str}] Сканирование завершено за {elapsed:.2f} сек. Сон {sleep_time:.2f} сек...\n")
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL ERROR] Исключение в основном потоке: {e}")
            time.sleep(10)


if __name__ == "__main__":
    api_key = BINGX_API_KEY
    api_secret = BINGX_SECRET
    
    # Создаем экземпляр BingxClient (для публичных запросов API ключи могут быть пустыми)
    client = BingxClient(api_key, api_secret, testnet=True)
  
    try:
        main(client)
    except KeyboardInterrupt:
        print("\nСканер остановлен пользователем. До свидания!")
