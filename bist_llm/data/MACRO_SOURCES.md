# Macro data: sources and verification notes

Compiled 2026-11-25. Every value below was read from an official page opened during this session. Nothing was filled in from memory, and no value is left unconfirmed.

## cpi.csv (39 rows, 2023-06 to 2026-08)

There was no earlier file, so all 39 rows are new.

- **Primary source:** the TÜİK news bulletin "Tüketici Fiyat Endeksi, <month year>" for each month, on TÜİK's data portal (`veriportali.tuik.gov.tr/tr/press/<Sayı>`). I opened all 39 bulletins.
  - `publication_date` is the bulletin's "Yayım Tarihi". Every bulletin was released at 10:00.
  - `cpi_mom_pct` and `cpi_yoy_pct` are the headline monthly and annual changes ("bir önceki aya göre" and "bir önceki yılın aynı ayına göre").
- **Cross-check 1 (values):** all 78 values match the TCMB table at İstatistikler > Enflasyon Verileri > Tüketici Fiyatları. The script that wrote the CSV asserts this.
- **Cross-check 2 (dates):** all 39 publication dates match the "Yayımlananlar" (already published) entries in the national release calendar at `www.tuik.gov.tr/Kurumsal/Veri_Takvimi` (years 2023 to 2026).
- **Releases not on the 3rd of the month:**
  - 2023-06 → 2023-07-05
  - 2023-08 → 2023-09-04
  - 2023-11 → 2023-12-04
  - 2024-01 → 2024-02-05
  - 2024-02 → 2024-03-04
  - 2024-07 → 2024-08-05
  - 2024-10 → 2024-11-04
  - 2025-04 → 2025-05-05
  - 2025-07 → 2025-08-04
  - 2025-12 → 2026-01-05
  - 2026-04 → 2026-05-04
  - 2026-05 → 2026-06-05

  Most of these are the 3rd moved off a weekend. Two are not: 2023-07-05 (the 3rd was a Monday) and 2026-06-05 (the 3rd was a Wednesday). The bulletin and the release calendar both show these dates, and the bulletins do not give a reason.
- **Base-year change:**
  - The bulletins up to and including December 2025 use TÜFE (2003=100).
  - From the January 2026 bulletin (published 2026-02-03, Sayı 58293), TÜİK uses TÜFE (2025=100) with the COICOP 2018 classification (174 sub-classes).
  - Rates published before the change were not revised. The January 2026 bulletin repeats the January 2025 rates (5.03 / 42.12) and the January 2024 rates (6.70 / 64.86) unchanged.
  - The `source` column gives the base for each row.
- **Two equal monthly rates:** the monthly change was 1.84 in both 2026-08 (Sayı 58290) and 2026-09 (Sayı 58298, outside the range). Both bulletins confirm this; it is not a copying error.
- **Bulletin numbers:** "Sayı" is the bulletin number shown on the current TÜİK portal. Bulletins up to November 2025 keep their 49xxx, 53xxx and 54xxx numbers. December 2025 and later use 58xxx numbers.

## policy_rate.csv (13 rows)

- **Result:** I checked all 13 current rows against TCMB's PPK (Monetary Policy Committee) decision announcements ("Faiz Oranlarına İlişkin Basın Duyurusu"). All 13 are correct, so no dates or values changed.
- **No missing changes:** I read every PPK decision from 2023-12-21 to 2026-09-10. TCMB's PPK lists for 2023 to 2026 contain no other rate change.
- **2026 meetings:**
  - 22 Jan: cut from 38 to 37 (2026-01).
  - 12 Mar (2026-12), 22 Apr (2026-17), 11 Jun (2026-23), 23 Jul (2026-28) and 10 Sep (2026-38): all held at 37.
- **The 2026-09-10 row:** kept. A decision was announced that day (Basın Duyurusu 2026-38): a hold at 37%.
- **Next change (outside the range):** on 22 Oct 2026, TCMB cut from 37 to 36 (2026-45).
- **Unscheduled meeting:** TCMB's 2025 list marks only 20 Mar 2025 (2025-20) as an interim meeting ("Ara Toplantı").
  - That meeting raised only the overnight lending rate (to 46%). The one-week repo rate stayed at 42.5, so it has no row.
  - 17 Apr 2025 (2025-24) is not marked as an interim meeting. That is the decision that raised the policy rate from 42.5 to 46.
- **Overnight corridor (not the policy rate):**
  - TCMB stopped one-week repo auctions on 1 Mar 2026 (Basın Duyurusu 2026-11) and restarted them on 23 Aug 2026 (2026-35).
  - In that period, every PPK statement kept the policy rate (one-week repo) at 37, with the overnight lending rate at 40 and the overnight borrowing rate at 35.5.
  - The file records only the one-week repo rate.
- **Dates:** `effective_date` is the decision date, as you asked. TCMB's own "1 Hafta Repo" table dates each change one day later (for example 22.12.2023, 23.01.2026).
- **Rate before the range:** the rate in force on 2023-12-01 was 40.00. It was set on 2023-11-23 and is shown on the "1 Hafta Repo" table as 24.11.2023. It has no row because it is before the range.

## fed_rate.csv (7 rows)

- **Result:** I checked all 7 current rows against the FOMC statements on federalreserve.gov (`monetaryYYYYMMDDa.htm`). All 7 are correct, so nothing changed.
- **2026 so far:** there was no target-range change between 2026-01-01 and 2026-09-10. The statements of 28 Jan, 18 Mar, 29 Apr, 17 Jun and 29 Jul 2026 all keep the range at 3-1/2 to 3-3/4 percent.
- **No other changes:** the FOMC statements from 2023-07-26 onward and the Fed's open-market rate table show no other change in the range, including no change between meetings.
- **Next change (outside the range):** the 16 Sep 2026 statement **raised** the range by 1/4 point to 3.75–4.00 (12–0 vote). It has no row because it is after 2026-09-10.
- **Dates:** `effective_date` is the FOMC statement (decision) date, as in your current file. The Fed's table at `federalreserve.gov/monetarypolicy/openmarket.htm` dates each change the following day (for example 2024-11-08, 2025-12-11).

## Sources used

- **TÜİK** (Turkish statistics office):
  - Bulletin pages `https://veriportali.tuik.gov.tr/tr/press/<Sayı>` (39 CPI bulletins, plus September and October 2026).
  - National release calendar: `https://www.tuik.gov.tr/Kurumsal/Veri_Takvimi`
- **TCMB** (Turkish central bank):
  - CPI table: `https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Istatistikler/Enflasyon+Verileri/Tuketici+Fiyatlari`
  - PPK decision lists: `.../Temel+Faaliyetler/Para+Politikasi/PPK/2023` to `/2026`
  - Press releases: `.../duyurular/basin/<year>/duy<no>`
  - One-week repo history: `.../Merkez+Bankasi+Faiz+Oranlari/1+Hafta+Repo`
- **Federal Reserve:**
  - FOMC statements: `https://www.federalreserve.gov/newsevents/pressreleases/monetary<date>a.htm`
  - Meeting calendar: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
  - Rate-change table: `https://www.federalreserve.gov/monetarypolicy/openmarket.htm`
