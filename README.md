# NEUTRONS

NEUTRONS là hệ thống trợ lý AI cá nhân chạy cục bộ, được xây dựng theo mô hình modular và local-first. Dự án hiện đang ở giai đoạn khởi tạo: backend API, lớp mô hình cục bộ qua Ollama, và bộ orchestrator để xử lý nhiệm vụ cơ bản.

## Mục tiêu kiến trúc

- Local-first: không phụ thuộc vào nền tảng đám mây cho thời điểm chạy chính.
- Một mô hình cục bộ: hiện tại dùng Ollama với `qwen3:4b`.
- Tách rõ trách nhiệm: cấu hình, API, model gateway, orchestration.
- Dễ mở rộng: thêm module chuyên môn trong tương lai mà không phá vỡ thiết kế.

## Kiểm tra môi trường hiện tại

Environment đã được xác minh với:

```powershell
echo %OLLAMA_MODELS%
ollama list
curl http://127.0.0.1:11434/api/tags
```

Kết quả hợp lệ phải trỏ đến:

```text
E:\OllamaModels\models
```

Hiện tại model `qwen3:4b` đã tồn tại và sẵn sàng dùng.

## Chạy giao diện desktop trên Windows

Đảm bảo Ollama đang chạy và model `qwen3:4b` đã có. Mở thư mục dự án rồi:

1. Cài đặt các dependency (chỉ cần làm lần đầu):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Nhấp đúp `start_neutrons.bat`, hoặc chạy trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m neutrons.desktop
```

Ứng dụng sẽ tự khởi động backend trên `127.0.0.1:8000` nếu backend chưa chạy,
sau đó mở cửa sổ chat. Khi đóng UI, backend do UI khởi chạy cũng sẽ được dừng.
Nếu backend NEUTRONS đã chạy sẵn, ứng dụng sẽ kết nối vào đó và không dừng server
đó khi đóng cửa sổ.

Giao diện gửi tin nhắn qua API cục bộ; phản hồi model có thể mất một lúc, tùy
hiệu năng máy. `Enter` gửi tin nhắn và `Shift+Enter` xuống dòng.
Trên máy chỉ dùng CPU, một câu trả lời ngắn vẫn có thể mất vài chục giây. Model
Qwen3 được giới hạn tối đa 512 token mỗi lượt để tránh sinh suy nghĩ quá dài và
vượt thời gian chờ của backend.

## Chạy backend riêng (tùy chọn)

1. Tạo môi trường ảo nếu cần.
2. Cài đặt dependency:

```powershell
python -m pip install -r requirements.txt
```

3. Khởi động FastAPI:

```powershell
python -m uvicorn neutrons.app:app --reload --host 127.0.0.1 --port 8000
```

4. Kiểm tra sức khỏe API:

```powershell
curl http://127.0.0.1:8000/healthz
```

## API

### GET /healthz

Trả về trạng thái hệ thống.

### POST /api/v1/chat

Body mẫu:

```json
{
  "message": "Xin chào NEUTRONS, hãy tóm tắt kế hoạch công việc của tôi."
}
```

Khi server đang chạy, hệ thống sẽ gọi Ollama local và trả về câu trả lời mô hình đã cấu hình.

## Lưu ý kỹ thuật

- NEUTRONS không tích hợp thêm provider ngoài Ollama ở giai đoạn hiện tại.
- Tất cả mô hình inference đều đi qua một lớp giao diện duy nhất để dễ thay đổi sau này.
- Nếu `ollama serve` đang ở port `11434`, không khởi động server thứ hai.

## Giai đoạn tiếp theo đề xuất

1. Tạo module lưu trữ memory và session.
2. Thêm bộ planner/task breakdown.
3. Tạo module tools cho file và hệ thống.
4. Mở rộng UI dashboard theo nhu cầu.
