# Veri dosyaları (fundamentals.csv: hocanın KAP rapor PDFlerinden parse_financials.py ile üretildi)
- fundamentals.csv: disclosure_date = KAP'ta bilançonun açıklandığı tarih (YYYY-MM-DD), çeyrek sonu DEĞİL.
  revenue/ebitda/net_profit = o çeyreğin (3 aylık) değeri, net_debt = çeyrek sonu net borç,
  ebitda_ltm = son 12 ay FAVÖK (Net Borç/FAVÖK için). Aynı birim (TL bin ya da milyon) kullan.
  Başka hisse için aynı formatta satır ekle (EREGL, THYAO, ASELS ...).
- policy_rate.csv: TCMB bir hafta repo faizi karar tarihleri (effective_date, policy_rate_pct).
- cpi.csv: TÜİK enflasyon, publication_date = açıklanma tarihi.
Boş bırakılan alan araçlarda "veri yok" uyarısı üretir; uydurma değer girme.
