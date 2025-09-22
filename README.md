# BTC Machine (Modüler)

Bu depo, 3 katmanlı (Veri Toplama → Analiz/Özellikler → Karar & Backtest) bir **BTC makinesinin** temiz, genişletilebilir iskeletini içerir.

## Hızlı Başlangıç

```bash
# Ortamı (ör. conda) aktifleştirin ve bağımlılıkları yükleyin
pip install -r requirements.txt

# 1) Verileri indir
python main.py download --config ./config/settings.yaml

# 2) Modeli eğit (opsiyonel)
python main.py train --config ./config/settings.yaml

# 3) Backtest çalıştır
python main.py backtest --config ./config/settings.yaml
```

- Veriler `./data/` klasörüne CSV olarak kaydedilir.
- Model `./models/` klasörüne `btc_gb.joblib` olarak yazılır.
- Backtest öz sermaye eğrisi `./data/backtest_equity.csv` dosyasına kaydedilir.

## Katmanlar

- `src/data_aggregator.py`: Binance (ccxt) OHLCV, Yahoo (QQQ, DXY), RSS + VADER duygu, Funding & OI (Binance Futures) çekicileri.
- `src/features.py`: RSI, SMA, QQQ trendi, BTC↔QQQ korelasyonu, FNG vb.
- `src/decision.py`: Kural tabanlı sinyal + (varsa) ML ile ensemble.
- `src/ml_model.py`: Gradient Boosting ile basit sınıflandırıcı eğitim/servis.
- `src/macro_calendar.py`: ICS takviminden makro olayları çekip risk penceresi kontrolü.
- `src/backtest.py`: Basit, vektörize backtest (ücret + kayma + hedef pozisyon).

## Yapılandırma

`config/settings.yaml` içinde:
- Semboller, zaman dilimi, lookback günleri
- RSS kaynakları, duygu penceresi
- Makro ICS linkleri ve risk penceresi
- Kurallar (RSI eşikleri, min duygu), risk parametreleri (fee, slippage, max pozisyon)

## Notlar

- Toplam piyasa değeri & BTC dominansı **canlı snapshot** olarak kolay; **tarihsel** seri gerekiyorsa CoinGecko/Messari gibi API'lere bağlamak için `data_aggregator.py` içine kolayca yeni fonksiyon ekleyin.
- Twitter (X) ve NewsAPI anahtarları opsiyonel. RSS ile out-of-the-box çalışır.
- OI/Funding için güvenli pencereleme ve 400 hatasına karşı otomatik dar aralık stratejisi eklendi.
- `backtest.py` basit bir portföy simülatörüdür; ATR stop, TP, trailing gibi gelişmişler için genişletme noktaları bırakıldı.
