import * as ort from 'onnxruntime-web';
import { nms, estimateSimilarityTransform, INSIGHTFACE_DST_PTS } from './utils';
import type { Point } from './utils';

/**
 * Pipeline xử lý khuôn mặt (Detection & Recognition) sử dụng ONNX Runtime Web.
 * Tối ưu hóa hiệu suất thông qua việc tái sử dụng bộ nhớ Buffer Canvas.
 */
export class InsightFace {
    private detSess: ort.InferenceSession | null = null;
    private recSess: ort.InferenceSession | null = null;
    
    // Hệ thống Buffer Canvas giúp giảm thiểu Garbage Collection và cấp phát bộ nhớ liên tục.
    private stageCanvas = document.createElement('canvas'); // Xử lý chuẩn hóa ảnh gốc.
    private detCanvas = document.createElement('canvas');   // Buffer đầu vào cho model Detection (640x640).
    private alignCanvas = document.createElement('canvas'); // Buffer cho quá trình Face Alignment (112x112).

    constructor() {
        this.detCanvas.width = this.detCanvas.height = 640;
        this.alignCanvas.width = this.alignCanvas.height = 112;
    }

    /**
     * Khởi tạo Inference Session với cấu hình tối ưu hóa đồ thị và WebAssembly Backend.
     */
    async load(detPath: string, recPath: string) {
        const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' } as any;
        this.detSess = await ort.InferenceSession.create(detPath, opt);
        this.recSess = await ort.InferenceSession.create(recPath, opt);
    }

    /**
     * Loại bỏ các sai lệch về hướng (orientation) từ metadata EXIF và tọa độ hiển thị CSS.
     */
    private flattenImage(img: HTMLImageElement): HTMLCanvasElement {
        this.stageCanvas.width = img.naturalWidth;
        this.stageCanvas.height = img.naturalHeight;
        const ctx = this.stageCanvas.getContext('2d', { willReadFrequently: true })!;
        ctx.drawImage(img, 0, 0);
        return this.stageCanvas;
    }

    /**
     * Thực hiện quy trình Detection: Letterboxing -> Normalization -> Inference -> Decoding -> NMS.
     */
    async analyze(imgElement: HTMLImageElement) {
        const img = this.flattenImage(imgElement);
        const w = img.width, h = img.height;
        
        // Tính toán tỷ lệ scale để fit ảnh vào khung 640x640 mà không làm biến dạng (Letterbox).
        const scale = Math.min(640 / w, 640 / h);
        const nw = w * scale, nh = h * scale;
        const dw = (640 - nw) / 2, dh = (640 - nh) / 2;

        const ctx = this.detCanvas.getContext('2d')!;
        ctx.fillStyle = '#808080'; // Sử dụng hằng số màu xám cho vùng padding.
        ctx.fillRect(0, 0, 640, 640);
        ctx.drawImage(img, dw, dh, nw, nh);

        // Chuyển đổi ImageData sang Tensor NCHW.
        // Thực hiện Normalization theo công thức: (x - 127.5) / 128.
        const data = ctx.getImageData(0, 0, 640, 640).data;
        const input = new Float32Array(3 * 640 * 640);
        for (let i = 0; i < 640 * 640; i++) {
            input[i] = (data[i * 4] - 127.5) / 128;     
            input[i + 409600] = (data[i * 4 + 1] - 127.5) / 128; 
            input[i + 819200] = (data[i * 4 + 2] - 127.5) / 128; 
        }

        const out = await this.detSess!.run({ [this.detSess!.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, 640, 640]) });
        const names = this.detSess!.outputNames;
        const strides = [8, 16, 32]; // Định nghĩa các stride tương ứng với các đầu ra của Feature Pyramid Network.
        let boxes: number[][] = [], scores: number[] = [], landmarks: number[][][] = [];

        // Giải mã dữ liệu từ tensor đầu ra dựa trên các strides và anchor points.
        for (let i = 0; i < 3; i++) {
            const stride = strides[i];
            const s = out[names[i]].data as Float32Array;
            const b = out[names[i+3]].data as Float32Array;
            const k = out[names[i+6]].data as Float32Array;
            const gw = 640 / stride;
            const numAnchors = s.length / (gw * gw);

            for (let j = 0; j < s.length; j++) {
                if (s[j] < 0.45) continue; // Lọc bỏ các anchors có độ tin cậy thấp.
                const cellIdx = Math.floor(j / numAnchors);
                const y = Math.floor(cellIdx / gw) * stride, x = (cellIdx % gw) * stride;
                
                // Quy đổi tọa độ từ tensor không gian 640x640 về không gian ảnh gốc.
                boxes.push([
                    (x - b[j*4]*stride - dw) / scale, (y - b[j*4+1]*stride - dh) / scale,
                    (x + b[j*4+2]*stride - dw) / scale, (y + b[j*4+3]*stride - dh) / scale
                ]);
                scores.push(s[j]);

                let kp: number[][] = [];
                for (let l = 0; l < 5; l++) {
                    kp.push([
                        (x + k[j*10 + l*2]*stride - dw) / scale,
                        (y + k[j*10 + l*2 + 1]*stride - dh) / scale
                    ]);
                }
                landmarks.push(kp);
            }
        }

        // Áp dụng thuật toán Non-Maximum Suppression để loại bỏ các box trùng lặp.
        const keep = nms(boxes, scores, 0.4);
        if (keep.length === 0) return null;

        // Ưu tiên chọn khuôn mặt có diện tích Bounding Box lớn nhất.
        let maxA = -1, best = keep[0];
        for (const idx of keep) {
            const a = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]);
            if (a > maxA) { maxA = a; best = idx; }
        }

        return { bbox: boxes[best], score: scores[best], kps: landmarks[best] };
    }

    /**
     * Thực hiện trích xuất Embedding: Similarity Transform (Alignment) -> BGR Normalization -> Recognition Inference.
     */
    async getEmbedding(kps: number[][]) {
        const img = this.stageCanvas; 
        const src: Point[] = kps.map(p => ({ x: p[0], y: p[1] }));
        
        // Tính toán ma trận Affine dựa trên các điểm Landmark tham chiếu để căn chỉnh khuôn mặt.
        const m = estimateSimilarityTransform(src, INSIGHTFACE_DST_PTS);

        const ctx = this.alignCanvas.getContext('2d')!;
        ctx.setTransform(1, 0, 0, 1, 0, 0); 
        ctx.clearRect(0, 0, 112, 112);
        
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        // Áp dụng ma trận biến đổi để Warp ảnh về chuẩn 112x112 (ArcFace Input).
        ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
        ctx.drawImage(img, 0, 0, img.width, img.height);
        ctx.restore();

        const data = ctx.getImageData(0, 0, 112, 112).data;
        const recIn = new Float32Array(3 * 112 * 112);
        // Hoán đổi kênh màu từ RGB sang BGR và Normalize cho model Recognition.
        for (let i = 0; i < 112 * 112; i++) {
            recIn[i] = (data[i * 4 + 2] - 127.5) / 128;         // Blue
            recIn[i + 12544] = (data[i * 4 + 1] - 127.5) / 128; // Green
            recIn[i + 25088] = (data[i * 4] - 127.5) / 128;     // Red
        }

        const out = await this.recSess!.run({ [this.recSess!.inputNames[0]]: new ort.Tensor('float32', recIn, [1, 3, 112, 112]) });
        return { 
            embedding: out[this.recSess!.outputNames[0]].data as Float32Array, 
            alignedCanvas: this.alignCanvas 
        };
    }
}