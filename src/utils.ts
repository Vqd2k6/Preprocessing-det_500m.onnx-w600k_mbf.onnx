export interface Point { x: number; y: number; }

/**
 * Tọa độ các điểm mốc (landmarks) chuẩn trong không gian 112x112.
 * Được sử dụng làm phân phối mục tiêu để căn chỉnh khuôn mặt về tư thế chính diện (Canonical Pose).
 */
export const INSIGHTFACE_DST_PTS: Point[] = [
    { x: 38.2946, y: 51.6963 }, { x: 73.5318, y: 51.5014 }, // Mắt trái, mắt phải
    { x: 56.0252, y: 71.7366 },                             // Đỉnh mũi
    { x: 41.5493, y: 92.3655 }, { x: 70.7299, y: 92.2041 }  // Khóe miệng trái, phải
];

/**
 * Ước tính ma trận biến đổi Similarity Transform (bao gồm tịnh tiến, quay và co giãn).
 * Sử dụng phương pháp bình phương tối thiểu để tối ưu hóa việc khớp các điểm nguồn vào điểm mục tiêu.
 */
export function estimateSimilarityTransform(src: Point[], dst: Point[]) {
    const n = src.length;
    let srcM = { x: 0, y: 0 }, dstM = { x: 0, y: 0 };
    
    // Tính toán trọng tâm (Centroid) của tập hợp điểm nguồn và điểm đích.
    for (let i = 0; i < n; i++) {
        srcM.x += src[i].x; srcM.y += src[i].y;
        dstM.x += dst[i].x; dstM.y += dst[i].y;
    }
    srcM.x /= n; srcM.y /= n; dstM.x /= n; dstM.y /= n;

    let srcVar = 0, cos = 0, sin = 0;
    // Tính toán phương sai và hiệp phương sai để xác định tỷ lệ co giãn và góc quay.
    for (let i = 0; i < n; i++) {
        const sX = src[i].x - srcM.x, sY = src[i].y - srcM.y;
        const dX = dst[i].x - dstM.x, dY = dst[i].y - dstM.y;
        srcVar += (sX * sX + sY * sY);
        cos += (sX * dX + sY * dY);
        sin += (sX * dY - sY * dX);
    }

    // Xác định hệ số co giãn (scale) và các thành phần ma trận quay (a, b).
    const scale = srcVar === 0 ? 1 : Math.sqrt(cos * cos + sin * sin) / srcVar;
    const angle = Math.atan2(sin, cos);
    const a = scale * Math.cos(angle);
    const b = scale * Math.sin(angle);

    // Giải phương trình tìm vector tịnh tiến (tx, ty) để hoàn tất ma trận Affine.
    const tx = dstM.x - (a * srcM.x - b * srcM.y);
    const ty = dstM.y - (b * srcM.x + a * srcM.y);

    // Trả về ma trận dưới dạng mảng tương thích với phương thức setTransform của Canvas API.
    return [a, b, -b, a, tx, ty];
}

/**
 * Thuật toán Non-Maximum Suppression (NMS) để lọc các vùng nhận diện chồng lấn.
 * Sử dụng chỉ số Intersection over Union (IoU) để loại bỏ các ứng viên có độ tin cậy thấp hơn.
 */
export function nms(boxes: number[][], scores: number[], threshold: number): number[] {
    // Sắp xếp các chỉ số theo thứ tự giảm dần của điểm số tin cậy (Confidence Score).
    const indices = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s).map(v => v.i);
    const keep: number[] = [];
    const ignored = new Set<number>();

    for (const i of indices) {
        if (ignored.has(i)) continue;
        keep.push(i);

        for (const j of indices) {
            if (i === j || ignored.has(j)) continue;
            
            // Tính toán diện tích vùng giao nhau (Intersection).
            const x1 = Math.max(boxes[i][0], boxes[j][0]), y1 = Math.max(boxes[i][1], boxes[j][1]);
            const x2 = Math.min(boxes[i][2], boxes[j][2]), y2 = Math.min(boxes[i][3], boxes[j][3]);
            const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            
            // Tính toán diện tích vùng hợp (Union).
            const areaI = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
            const areaJ = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]);
            
            // Loại bỏ ứng viên nếu tỷ lệ IoU vượt quá ngưỡng xác định.
            if (inter / (areaI + areaJ - inter) > threshold) ignored.add(j);
        }
    }
    return keep;
}