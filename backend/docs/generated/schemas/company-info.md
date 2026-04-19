# company.info

- Class: `CompanyReference`
- Method: `info`
- Signature: `(show_log = False) -> DataFrame chứa thông tin tổng quan công ty.`
- Return type: `DataFrame chứa thông tin tổng quan công ty.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.overview`

Get company info/overview.

## Purpose

Get company info/overview.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `omitted; default False` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `overview`

```text
business_model, symbol, founded_date, charter_capital, charter_capital_vnd, number_of_employees, listing_date, par_value, exchange, listing_price, listed_volume, ceo_name, ceo_position, inspector_name, inspector_position, establishment_license, business_code, tax_id, auditor, company_type, address, phone, fax, email, website, branches, history, outstanding_shares, as_of_date
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
business_model, symbol, founded_date, charter_capital, charter_capital_vnd, number_of_employees, listing_date, par_value, exchange, listing_price, listed_volume, ceo_name, ceo_position, inspector_name, inspector_position, establishment_license, business_code, tax_id, auditor, company_type, address, phone, fax, email, website, branches, history, outstanding_shares, as_of_date
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-17T05:26:47.721590+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, sector, industry, profile, history, charter_capital, issued_share
```
- Dtypes: `{'symbol': 'str', 'sector': 'str', 'industry': 'str', 'profile': 'str', 'history': 'str', 'charter_capital': 'int64', 'issued_share': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "sector": "Ngân hàng",
    "industry": "Ngân hàng",
    "profile": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam (Vietcombank) chính thức đi vào hoạt động ngày 01/04/1963. Là ngân hàng thương mại nhà nước đầu tiên được Chính phủ lựa chọn thực hiện thí điểm cổ phần hoá, Ngân hàng Ngoại thương Việt Nam chính thức hoạt động với tư cách là một Ngân hàng Thương mại Cổ phần từ ngày 02/06/2008 sau khi thực hiện thành công kế hoạch cổ phần hóa thông qua việc phát hành cổ phiếu lần đầu ra công chúng. Năm 2024, so với cùng kỳ, biên lãi thuần (NIM) ở mức 2.86%, giảm 0.15%. Tỷ lệ nợ xấu ở mức 0.96%, giảm 0.02%. Tỷ lệ bao phủ nợ xấu ở mức 223.31%, giảm 6.99%. Lợi nhuận sau thuế công ty mẹ có giá trị bằng 33,8 nghìn tỷ đồng, tăng 2.42%. Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE) ở mức 18.74%, giảm 3.25%. VCB chính thức niêm yết và giao dịch trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh từ năm 2009.",
    "history": " - Ngày 30/10/1962: Ngân hàng Ngoại thương Việt Nam (Vietcombank) được thành lập có tiền thân là Cục Ngoại Hối trực thuộc Ngân Hàng Quốc Gia Việt Nam;  - Ngày 01/04/1963: Vietcombank chính thức đi vào hoạt động;  - Năm 1990: Vietcombank chuyển thành một NHTM nhà nước hoạt động đa năng;  - Năm 2007: Vietcombank và NHTMCP SeaBank ký kết Hợp đồng với đối tác Cardif thành lập Công ty TNHH Bảo hiểm Nhân thọ Vietcombank – Cardif (VCLI);  - Ngày 26/12/2007: Vietcombank phát hành đợt cổ phiếu đầu tiên ra công chúng.  - Năm 2008: Ngân hàng ngoại thương Việt Nam chính thức chuyển đổi thành Ngân Hàng TMCP Ngoại Thương Việt Nam;  - Năm 2009: Cổ phiếu của Vietcombank chính thức được niêm yết trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh (HOSE);  - Ngày 30/09/2011: Ngân hàng Mizuho (MHCB) đã chính thức trở thành nhà đầu tư chiến lược vào Vietcombank, nắm giữ 15% vốn điều lệ của Vietcombank;  - Ngày 16/01/2019: Tăng vốn điều lệ lên 37.088.774.480.000 đồng;  - Ngày 10/03/2022: Tăng vốn điều lệ lên 47.325.165.710.000 đồng;  - Ngày 05/10/2023: Tăng vốn điều lệ lên 55.890.912.620.000 đồng;  - Ngày 28/04/2025: Tăng vốn điều lệ lên 83.556.750.940.000 đồng do phát hành cổ phiếu trả cổ tức;",
    "charter_capital": 83556750940000,
    "issued_share": 8355675094
  }
]
```
