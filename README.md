
# Face Analysis Pipeline Documentation (Client-Side)

Hệ thống thực hiện quy trình phân tích khuôn mặt đầu cuối (End-to-End) dựa trên hai mô hình học sâu: **SCRFD (Face Detection)** và **ArcFace (Face Recognition)** thông qua ONNX Runtime Web. Quy trình được thiết kế để tối ưu hóa tài nguyên trên trình duyệt và đảm bảo độ chính xác của vector đặc trưng.

## 1. Kiến trúc Pipeline

Pipeline được chia thành hai giai đoạn chính:
1.  **Giai đoạn Detection:** Xác định vị trí khuôn mặt và 5 điểm mốc (landmarks).
2.  **Giai đoạn Recognition:** Căn chỉnh khuôn mặt (Alignment) và trích xuất vector đặc trưng (Embedding).

---

## 2. Giai đoạn 1: Face Detection (SCRFD)

### 2.1 Tiền xử lý dữ liệu (Pre-processing)
*   **Flatten Image:** Loại bỏ các thuộc tính định hướng (EXIF) từ tệp tin ảnh gốc bằng cách vẽ lại ảnh lên một Canvas trung gian, đảm bảo tọa độ pixel khớp với tọa độ hiển thị.
*   **Letterbox Resizing:** Resize ảnh về kích thước input chuẩn của mô hình (640x640). Để bảo toàn tỷ lệ khung hình (Aspect Ratio), ảnh được scale theo cạnh lớn nhất, các vùng trống còn lại được lấp đầy bằng hằng số màu xám (Padding).
*   **Normalization:** Dữ liệu pixel được chuyển đổi từ dải [0, 255] sang [-1, 1] theo công thức: 
    $$Input = (Pixel - 127.5) / 128$$
*   **NCHW Format:** Dữ liệu được tái cấu trúc thành Tensor 4 chiều [1, 3, 640, 640] trước khi đưa vào Inference Session.

### 2.2 Giải mã đầu ra đa tầng (Multi-stride Decoding)
Mô hình trả về 9 tensor đầu ra tương ứng với 3 tầng Feature Pyramid Network (FPN) có strides lần lượt là **8, 16, 32**. Tại mỗi tầng, quy trình giải mã bao gồm:
*   **Score Map:** Xác định xác suất có mặt khuôn mặt tại mỗi Anchor point. Chỉ các điểm có Score > 0.45 được giữ lại.
*   **Box Map:** Tính toán tọa độ Bounding Box bằng cách cộng các giá trị offset từ tâm Anchor point nhân với bước nhảy (stride).
*   **Landmark Map:** Giải mã 5 điểm mốc (mắt trái, mắt phải, mũi, khóe miệng trái/phải) tương tự như Bounding Box.
*   **Coordinate Projection:** Toàn bộ tọa độ được quy đổi ngược từ không gian 640x640 về không gian ảnh gốc dựa trên tỷ lệ scale và phần padding đã tính toán ở bước Letterbox.

### 2.3 Hậu xử lý (Post-processing)
*   **Non-Maximum Suppression (NMS):** Do mô hình có thể dự đoán nhiều box cho cùng một khuôn mặt tại các tầng khác nhau, thuật toán NMS được áp dụng với ngưỡng IoU 0.4 để loại bỏ các vùng chồng lấn, chỉ giữ lại các ứng viên có độ tin cậy cao nhất.
*   **Primary Face Selection:** Trong trường hợp có nhiều khuôn mặt, hệ thống sử dụng Heuristic tính diện tích (Area) để ưu tiên chọn khuôn mặt lớn nhất (thường là chủ thể chính).

---

## 3. Giai đoạn 2: Face Recognition (ArcFace)

### 3.1 Căn chỉnh khuôn mặt (Face Alignment)
Đây là bước quan trọng nhất để đảm bảo tính bất biến của vector đặc trưng đối với các tư thế quay mặt khác nhau:
*   **Similarity Transform:** Sử dụng 5 điểm Landmark tìm được từ Giai đoạn 1 để tính toán ma trận biến đổi Affine. Ma trận này khớp các điểm mốc thực tế vào một bộ khung điểm mốc chuẩn (**INSIGHTFACE_DST_PTS**) trong không gian 112x112.
*   **Warping:** Áp dụng ma trận biến đổi lên ảnh gốc thông qua Canvas API (`setTransform`), thực hiện xoay, co giãn và tịnh tiến để đưa khuôn mặt về tư thế chính diện (Canonical View).

### 3.2 Feature Extraction (Inference)
*   **Color Space Conversion:** Model ArcFace yêu cầu đầu vào theo định dạng màu **BGR**. Hệ thống thực hiện hoán đổi kênh R và B từ ImageData của trình duyệt trước khi tạo Tensor.
*   **Input Shape:** Dữ liệu đầu vào là Tensor [1, 3, 112, 112].
*   **Embedding Generation:** Kết quả đầu ra là một vector 512 chiều (L2-Normalized Embedding). Vector này đại diện cho danh tính của khuôn mặt và được sử dụng cho các bài toán so khớp (Face Matching) hoặc tìm kiếm (Face Search).

---

## 4. Tối ưu hóa MLOps Edge-Computing
*   **Memory Management:** Sử dụng các Canvas đối tượng tĩnh (`stageCanvas`, `detCanvas`, `alignCanvas`) để tái sử dụng bộ nhớ, tránh việc trình duyệt thực hiện Garbage Collection liên tục gây giật lag (stuttering).
*   **Execution Backend:** Cấu hình `executionProviders: ['wasm']` kết hợp với `graphOptimizationLevel: 'all'` để tận dụng tối đa tập lệnh SIMD trên CPU thông qua WebAssembly, đảm bảo tốc độ thực thi tiệm cận môi trường Native.
*   **Asynchronous Processing:** Toàn bộ Pipeline được thiết kế bất đối xứng (async/await) để không gây nghẽn luồng xử lý giao diện (Main UI Thread).
