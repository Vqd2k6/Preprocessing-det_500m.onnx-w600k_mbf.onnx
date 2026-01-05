import { InsightFace } from './insightface';

// Khởi tạo các tham chiếu DOM Elements phục vụ giao diện người dùng và hiển thị kết quả.
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const originalPreview = document.getElementById('originalPreview') as HTMLImageElement;
const outputEl = document.getElementById('output') as HTMLElement;
const alignedCanvasDisplay = document.getElementById('alignedResult') as HTMLCanvasElement;

// Khởi tạo instance của engine phân tích khuôn mặt.
const faceEngine = new InsightFace();

/**
 * Quản lý vòng đời khởi tạo hệ thống: Tải trọng số mô hình và kích hoạt Runtime.
 * Ngăn chặn tương tác người dùng cho đến khi các Session ONNX sẵn sàng.
 */
async function init() {
    try {
        // Nạp các tệp tin trọng số đã được tối ưu hóa cho môi trường Web.
        await faceEngine.load('./public/models/det_500m.onnx', './public/models/w600k_mbf.onnx');
        fileInput.disabled = false;
    } catch (e) {
        console.error("Initialization Failed", e);
    }
}

/**
 * Xử lý sự kiện nạp dữ liệu hình ảnh từ Client.
 * Sử dụng Object URL để tối ưu hóa việc truyền dữ liệu vào phần tử Image mà không cần qua Server.
 */
fileInput.onchange = (e: any) => {
    const file = e.target.files[0];
    if (!file) return;
    originalPreview.src = URL.createObjectURL(file);
};

/**
 * Luồng thực thi chính của Pipeline khi dữ liệu hình ảnh đã sẵn sàng trong DOM.
 * Quy trình: Image Ingestion -> Face Detection -> Face Recognition -> Visualization.
 */
originalPreview.onload = async () => {
    outputEl.innerText = "Processing...";
    
    // Giai đoạn 1: Thực thi mô hình Detection để tìm kiếm tọa độ khuôn mặt và các điểm Landmark.
    const face = await faceEngine.analyze(originalPreview);

    if (!face) {
        outputEl.innerText = "No face detected.";
        return;
    }

    // Giai đoạn 2: Trích xuất đặc trưng (Feature Extraction) dựa trên các điểm Landmark đã tìm thấy.
    const result = await faceEngine.getEmbedding(face.kps);
    if (result) {
        // Đồng bộ hóa kết quả từ Buffer nội bộ của Engine ra Canvas hiển thị trên UI.
        const ctx = alignedCanvasDisplay.getContext('2d')!;
        ctx.clearRect(0, 0, 112, 112);
        ctx.drawImage(result.alignedCanvas, 0, 0);

        // Tổng hợp và định dạng các chỉ số kỹ thuật từ đầu ra của mô hình.
        outputEl.innerText = [
            `Score: ${(face.score * 100).toFixed(2)}%`, // Độ tin cậy của Detection.
            `BBox: [${face.bbox.map(Math.round).join(', ')}]`, // Tọa độ khung bao khuôn mặt.
            `Embedding (First 5):`, // Trích xuất 5 chiều đầu tiên của Feature Vector (512-d).
            `[${Array.from(result.embedding.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`
        ].join('\n');
    }
};

// Kích hoạt tiến trình khởi tạo hệ thống.
init();