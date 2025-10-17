# Crypto Price Tracker

**Crypto Price Tracker** เป็นเว็บแอปพลิเคชันที่เขียนด้วย **Streamlit + Python** สำหรับติดตามราคาสกุลเงินคริปโตแบบเรียลไทม์ วิเคราะห์แนวโน้ม และแจ้งเตือนเมื่อราคามีการเปลี่ยนแปลง

> ℹ️ โครงการนี้สร้างขึ้นเพื่อใช้ในการศึกษาและวิเคราะห์ข้อมูลเท่านั้น — ไม่ใช่คำแนะนำทางการเงิน

---

## 🎯 คุณสมบัติ (Features)

- **Market Overview**  
  แสดงราคาปัจจุบัน, การเปลี่ยนแปลงใน 24 ชั่วโมง & 7 วัน, มูลค่าตลาด (market cap), ปริมาณการซื้อขาย (volume) สำหรับเหรียญยอดนิยม  
- **Historical Data & 7-Day MA**  
  ดูประวัติราคาและเส้นค่าเฉลี่ยเคลื่อนที่ 7 วัน (Moving Average) ของเหรียญแต่ละตัว  
- **AI Trend Analyzer**  
  พยากรณ์ราคาในอนาคต (7 วัน) โดยใช้โมเดลเชิงเส้น (linear regression)  
- **Price Drop Alerts**  
  ตั้งค่าเกณฑ์ (threshold) เพื่อรับแจ้งเตือนเมื่อราคาลดลงเกินกว่าที่กำหนดจาก snapshot ที่บันทึกไว้  
- **AI Market Summary** *(Optional)*  
  ดึงสรุปภาพรวมตลาดด้วย OpenAI API (หากตั้งค่า API Key)  
- **UI ที่สวยงาม & ใช้งานง่าย**  
  อินเทอร์เฟซทันสมัย ตอบสนองดี (responsive)  

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)



```

.
├── app.py
├── requirements.txt
├── services/
│   ├── PriceFetcher.py
│   ├── TrendAnalyzer.py
│   ├── AlertEngine.py
│   ├── AISummarizer.py
│   └── DataLogger.py
└── README.md

````

- `app.py` — จุดเริ่มต้นของแอป Streamlit  
- `requirements.txt` — ไลบรารี/แพ็กเกจที่จำเป็นต้องติดตั้ง  
- โฟลเดอร์ `services/` — โมดูลแยกส่วนงาน (fetching, วิเคราะห์แนวโน้ม, การแจ้งเตือน, สรุปด้วย AI, บันทึกข้อมูล)  

---

## 🛠 วิธีติดตั้ง & รัน (Installation & Run)

1. โคลน (clone) โปรเจกต์นี้มา:
   ```bash
   git clone https://github.com/aarktik/Crypto-Price-Tracker.git
   cd Crypto-Price-Tracker
    ```

2. สร้าง virtual environment (แนะนำ):

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # บน Linux / macOS
   .\venv\Scripts\activate       # บน Windows
   ```

3. ติดตั้ง dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (ถ้าใช้ AI Summary) ตั้งค่า API Key สำหรับ OpenAI:

   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

5. รันแอป:

   ```bash
   streamlit run app.py
   ```

6. เปิดเว็บเบราว์เซอร์ไปที่ `http://localhost:8501`

---

## 🌐 Live Demo
👉 [Crypto Price Tracker (Render Deployment)](https://crypto-price-tracker-vqes.onrender.com/)

---

## Slide
🎨 Design Preview  [Crypto Price Tracker](https://www.canva.com/design/DAG1W8H460A/P9_mqKomq0dPTYmZ6hXNBg/edit?utm_content=DAG1W8H460A&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


---
