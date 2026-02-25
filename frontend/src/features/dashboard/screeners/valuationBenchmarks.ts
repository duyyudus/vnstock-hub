export interface SectorBenchmark {
    pe_good: number;
    pe_accept: number;
    pb_good: number;
    pb_accept: number;
    roe_good: number;
    roe_accept: number;
    roa_good: number;
    roa_accept: number;
    w_pe: number;
    w_pb: number;
    w_roe: number;
    w_roa: number;
    w_growth: number;
    w_stability: number;
    w_val: number;
    w_qual: number;
}

const entries: Array<[string, SectorBenchmark]> = [
    ['Ngân hàng', { pe_good: 8, pe_accept: 10, pb_good: 1.3, pb_accept: 2, roe_good: 0.2, roe_accept: 0.17, roa_good: 0.018, roa_accept: 0.013, w_pe: 0.6, w_pb: 0.4, w_roe: 0.4, w_roa: 0.15, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Bất động sản', { pe_good: 12, pe_accept: 20, pb_good: 1.5, pb_accept: 3, roe_good: 0.15, roe_accept: 0.1, roa_good: 0.06, roa_accept: 0.03, w_pe: 0.55, w_pb: 0.45, w_roe: 0.3, w_roa: 0.2, w_growth: 0.25, w_stability: 0.25, w_val: 0.45, w_qual: 0.55 }],
    ['Phần mềm & Dịch vụ Máy tính', { pe_good: 18, pe_accept: 25, pb_good: 3.5, pb_accept: 5, roe_good: 0.25, roe_accept: 0.15, roa_good: 0.1, roa_accept: 0.06, w_pe: 0.65, w_pb: 0.35, w_roe: 0.3, w_roa: 0.2, w_growth: 0.35, w_stability: 0.15, w_val: 0.3, w_qual: 0.7 }],
    ['Kim loại', { pe_good: 10, pe_accept: 15, pb_good: 1, pb_accept: 1.5, roe_good: 0.12, roe_accept: 0.08, roa_good: 0.06, roa_accept: 0.04, w_pe: 0.45, w_pb: 0.55, w_roe: 0.3, w_roa: 0.2, w_growth: 0.25, w_stability: 0.25, w_val: 0.5, w_qual: 0.5 }],
    ['Hóa chất', { pe_good: 12, pe_accept: 18, pb_good: 1.5, pb_accept: 2.5, roe_good: 0.18, roe_accept: 0.12, roa_good: 0.1, roa_accept: 0.06, w_pe: 0.55, w_pb: 0.45, w_roe: 0.35, w_roa: 0.2, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Sản xuất thực phẩm', { pe_good: 16, pe_accept: 23, pb_good: 3, pb_accept: 5, roe_good: 0.25, roe_accept: 0.15, roa_good: 0.1, roa_accept: 0.05, w_pe: 0.6, w_pb: 0.4, w_roe: 0.35, w_roa: 0.15, w_growth: 0.3, w_stability: 0.2, w_val: 0.4, w_qual: 0.6 }],
    ['Du lịch & Giải trí', { pe_good: 15, pe_accept: 22, pb_good: 2, pb_accept: 4, roe_good: 0.12, roe_accept: 0.05, roa_good: 0.04, roa_accept: 0.015, w_pe: 0.65, w_pb: 0.35, w_roe: 0.25, w_roa: 0.2, w_growth: 0.35, w_stability: 0.2, w_val: 0.35, w_qual: 0.65 }],
    ['Bán lẻ', { pe_good: 18, pe_accept: 25, pb_good: 3, pb_accept: 5, roe_good: 0.22, roe_accept: 0.15, roa_good: 0.08, roa_accept: 0.05, w_pe: 0.7, w_pb: 0.3, w_roe: 0.3, w_roa: 0.2, w_growth: 0.35, w_stability: 0.15, w_val: 0.35, w_qual: 0.65 }],
    ['Dịch vụ tài chính', { pe_good: 12, pe_accept: 17, pb_good: 1.5, pb_accept: 2.5, roe_good: 0.15, roe_accept: 0.1, roa_good: 0.04, roa_accept: 0.02, w_pe: 0.6, w_pb: 0.4, w_roe: 0.35, w_roa: 0.15, w_growth: 0.3, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Sản xuất Dầu khí', { pe_good: 15, pe_accept: 22, pb_good: 1.5, pb_accept: 2.5, roe_good: 0.12, roe_accept: 0.08, roa_good: 0.04, roa_accept: 0.02, w_pe: 0.6, w_pb: 0.4, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Bia và đồ uống', { pe_good: 15, pe_accept: 20, pb_good: 2.5, pb_accept: 4, roe_good: 0.2, roe_accept: 0.15, roa_good: 0.12, roa_accept: 0.08, w_pe: 0.6, w_pb: 0.4, w_roe: 0.35, w_roa: 0.2, w_growth: 0.25, w_stability: 0.2, w_val: 0.45, w_qual: 0.55 }],
    ['Nước & Khí đốt', { pe_good: 18, pe_accept: 23, pb_good: 2.5, pb_accept: 4, roe_good: 0.18, roe_accept: 0.12, roa_good: 0.1, roa_accept: 0.06, w_pe: 0.55, w_pb: 0.45, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.45, w_qual: 0.55 }],
    ['Bảo hiểm nhân thọ', { pe_good: 12, pe_accept: 18, pb_good: 1.2, pb_accept: 2, roe_good: 0.14, roe_accept: 0.08, roa_good: 0.025, roa_accept: 0.012, w_pe: 0.6, w_pb: 0.4, w_roe: 0.35, w_roa: 0.15, w_growth: 0.3, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Dược phẩm', { pe_good: 14, pe_accept: 20, pb_good: 2, pb_accept: 3, roe_good: 0.2, roe_accept: 0.12, roa_good: 0.1, roa_accept: 0.06, w_pe: 0.6, w_pb: 0.4, w_roe: 0.35, w_roa: 0.2, w_growth: 0.25, w_stability: 0.2, w_val: 0.45, w_qual: 0.55 }],
    ['Hàng cá nhân', { pe_good: 16, pe_accept: 23, pb_good: 3, pb_accept: 5, roe_good: 0.24, roe_accept: 0.15, roa_good: 0.1, roa_accept: 0.06, w_pe: 0.6, w_pb: 0.4, w_roe: 0.3, w_roa: 0.2, w_growth: 0.3, w_stability: 0.2, w_val: 0.4, w_qual: 0.6 }],
    ['Sản xuất & Phân phối Điện', { pe_good: 12, pe_accept: 18, pb_good: 1.2, pb_accept: 2, roe_good: 0.14, roe_accept: 0.08, roa_good: 0.06, roa_accept: 0.03, w_pe: 0.55, w_pb: 0.45, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Thiết bị, Dịch vụ và Phân phối Dầu khí', { pe_good: 12, pe_accept: 18, pb_good: 1.2, pb_accept: 2, roe_good: 0.12, roe_accept: 0.06, roa_good: 0.04, roa_accept: 0.02, w_pe: 0.55, w_pb: 0.45, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Vận tải', { pe_good: 13, pe_accept: 18, pb_good: 1.5, pb_accept: 2.5, roe_good: 0.16, roe_accept: 0.1, roa_good: 0.07, roa_accept: 0.04, w_pe: 0.55, w_pb: 0.45, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Xây dựng và Vật liệu', { pe_good: 10, pe_accept: 16, pb_good: 1.2, pb_accept: 2, roe_good: 0.14, roe_accept: 0.08, roa_good: 0.06, roa_accept: 0.03, w_pe: 0.5, w_pb: 0.5, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
    ['Điện tử & Thiết bị điện', { pe_good: 12, pe_accept: 18, pb_good: 1.5, pb_accept: 2.5, roe_good: 0.14, roe_accept: 0.08, roa_good: 0.06, roa_accept: 0.03, w_pe: 0.55, w_pb: 0.45, w_roe: 0.3, w_roa: 0.25, w_growth: 0.25, w_stability: 0.2, w_val: 0.5, w_qual: 0.5 }],
];

export const DEFAULT_SECTOR_BENCHMARK: SectorBenchmark = {
    pe_good: 12,
    pe_accept: 18,
    pb_good: 1.5,
    pb_accept: 2.5,
    roe_good: 0.14,
    roe_accept: 0.1,
    roa_good: 0.06,
    roa_accept: 0.03,
    w_pe: 0.55,
    w_pb: 0.45,
    w_roe: 0.3,
    w_roa: 0.2,
    w_growth: 0.3,
    w_stability: 0.2,
    w_val: 0.5,
    w_qual: 0.5,
};

export const VALUATION_SECTOR_BENCHMARKS: ReadonlyArray<Readonly<[string, SectorBenchmark]>> = entries;
