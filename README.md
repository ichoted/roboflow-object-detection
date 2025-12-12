# Roboflow Object Detection Web App

Web Application สำหรับตรวจจับวัตถุ (Object Detection) โดยใช้ Roboflow Inference SDK และ FastAPI

## Requirements

- Python 3.9+
- [Roboflow Inference Server](https://github.com/roboflow/inference)

## Setup Project

1. **Clone Project หรือเตรียมโฟลเดอร์**

2. **สร้าง Virtual Environment (แนะนำ)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # สำหรับ macOS/Linux
   # venv\Scripts\activate  # สำหรับ Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## การรัน Roboflow Inference Server

ก่อนใช้งานเว็บ ต้องรัน Inference Server ก่อน (Local):

```bash
/Users/paphawara/Library/Python/3.9/bin/inference server start
```

โปรเจกต์นี้ตั้งค่าให้เชื่อมต่อกับ `http://localhost:9001`

## การรัน Web Application

รันคำสั่งต่อไปนี้เพื่อเปิด Web Server:

```bash
/Users/paphawara/Library/Python/3.9/bin/uvicorn main:app --reload --port 8000
```

## การใช้งาน

1. เปิด Browser ไปที่ [http://localhost:8000](http://localhost:8000)
2. คลิกที่พื้นที่อัปโหลด หรือลากไฟล์รูปภาพมาใส่
3. กดปุ่ม **Analyze Image**
4. ผลลัพธ์ (JSON) จะแสดงทางด้านขวา

## โครงสร้างไฟล์

- `main.py`: โค้ดส่วน Backend (FastAPI) เชื่อมต่อกับ Inference SDK
- `templates/index.html`: หน้าจอ User Interface
- `requirements.txt`: รายชื่อ Library ที่ต้องใช้
